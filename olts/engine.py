"""OnlineTSF kernels with one causal scheduler and shared masked scoring.

This is the main study's engine. Upstream OnlineTSF files are unmodified (git submodule
third_party/OnlineTSF, pinned to commit 0a86461). Changes are confined to device acquisition,
data arrival scheduling, masked loss, and explicit state/provenance capture.
"""
import os
# torch.use_deterministic_algorithms(True) needs this for cuBLAS; set before CUDA starts.
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import copy
import importlib
import random
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from .common import UPSTREAM, digest

if not (UPSTREAM / 'exp' / 'exp_online.py').is_file():
    raise ImportError(f'OnlineTSF not found at {UPSTREAM}; run: git submodule update --init')
sys.path.insert(0, str(UPSTREAM))
from exp.exp_online import Exp_Online, Exp_ER
from exp.exp_proceed import Exp_Proceed


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)


def arguments(model, dataset, W, H, channels, lr, device):
    large = dataset in ('Weather', 'Electricity', 'ETTm1', 'ETTm2')
    ett = dataset in ('ETTh1', 'ETTh2')
    return SimpleNamespace(model=model, dataset=dataset, seq_len=W, pred_len=H,
        enc_in=channels, individual=False, learning_rate=lr, online_learning_rate=lr,
        use_gpu=device.startswith('cuda'), gpu=0, use_multi_gpu=False, local_rank=-1,
        pin_gpu=True, normalization=None, compile=False, freeze=False, use_amp=False,
        save_opt=True, optim='Adam', wrap_data_class=[], e_layers=3,
        n_heads=4 if ett else 16, d_model=16 if ett else (128 if large else 64),
        d_ff=128 if ett or not large else 256, dropout=.3 if ett else .2,
        fc_dropout=.3 if ett else .2, head_dropout=0., patch_len=16, stride=8,
        padding_patch='end', revin=True, affine=False, subtract_last=False,
        decomposition=False, kernel_size=25, merge_weights=1, tune_mode='down_up',
        act='sigmoid', concept_dim=200, bottleneck_dim=32, individual_generator=False,
        ema=0., wo_clip=False, do_predict=False, joint_update_valid=False,
        val_online_lr=True, online_method='Online')


class SafeInterface:
    def _acquire_device(self):
        # The upstream method resets CUDA_VISIBLE_DEVICES; never call it in Slurm.
        return torch.device('cuda:0' if self.args.use_gpu else 'cpu')

    def _process_batch(self, batch):
        # Forecasts cannot receive labels or masks as decoder inputs.
        return [batch[0]]


class Naive(SafeInterface, Exp_Online):
    pass


class ER(SafeInterface, Exp_ER):
    pass


class PROCEED(SafeInterface, Exp_Proceed):
    def _update(self, batch, criterion, optimizer, scaler=None):
        result = super()._update(batch, criterion, optimizer, scaler)
        # Loss uses observed targets only. Native recent state receives the same
        # causally filled observations as every model input, never NaN labels.
        self.model.recent_batch = torch.cat((batch[0], batch[3]), dim=-2)
        return result

    def _update_online(self, batch, criterion, optimizer, scaler=None, flag_current=False):
        result = super()._update_online(batch, criterion, optimizer, scaler, flag_current)
        self.model.recent_batch = torch.cat((batch[0], batch[3]), dim=-2)
        return result


def make(args, method='Naive'):
    return {'Naive': Naive, 'ER': ER, 'PROCEED': PROCEED}[method](args)


def mse(pred, target):
    valid = torch.isfinite(target)
    if not bool(valid.any()):
        # A zero-connected loss is explicit; caller records skipped-observation batch.
        return pred.sum() * 0
    if not bool(torch.isfinite(pred).all()):
        raise FloatingPointError('Nonfinite prediction')
    return (pred[valid] - target[valid]).square().mean()


def batch(values, targets, origins, W, H, device):
    idx = np.asarray(origins, dtype=np.int64)
    x = np.stack([values[t-W+1:t+1] for t in idx])
    y = np.stack([targets[t+1:t+H+1] for t in idx])
    # Empty marker tensors preserve the native ER tuple/replay contract.
    marker = torch.empty((len(idx), 0), device=device)
    filled = np.stack([values[t+1:t+H+1] for t in idx])
    return (torch.as_tensor(x, device=device), torch.as_tensor(y, device=device), marker,
            torch.as_tensor(filled, device=device))


def prediction(exp, values, t, W):
    exp.model.eval()
    x = torch.as_tensor(values[t-W+1:t+1][None], device=exp.device)
    with torch.no_grad():
        out = exp.forward((x, None))
    if isinstance(out, (tuple, list)):
        out = out[0]
    if not bool(torch.isfinite(out).all()):
        raise FloatingPointError('Nonfinite forecast')
    return out[0].detach().cpu().numpy().copy()


def cpu_tree(obj):
    if torch.is_tensor(obj):
        return obj.detach().cpu().clone()
    if isinstance(obj, dict):
        return {k: cpu_tree(v) for k, v in obj.items()}
    if isinstance(obj, (tuple, list)):
        return type(obj)(cpu_tree(v) for v in obj)
    return copy.deepcopy(obj)


def snapshot(exp):
    model = exp.model
    r = dict(model=cpu_tree(model.state_dict()), optimizer=cpu_tree(exp.model_optim.state_dict()),
        # PROCEED recent_batch is deliberately non-persistent in native state_dict.
        buffers={n: cpu_tree(v) for n, v in model.named_buffers()},
        requires_grad={n: p.requires_grad for n, p in model.named_parameters()},
        flags={n: getattr(model, n) for n in ('flag_update','flag_current','flag_basic','flag_online_learning') if hasattr(model,n)},
        rng_python=random.getstate(), rng_numpy=np.random.get_state(), rng_torch=torch.get_rng_state(),
        rng_cuda=torch.cuda.get_rng_state_all() if exp.device.type == 'cuda' else [])
    if hasattr(exp, 'buffer'):
        r['replay'] = cpu_tree(exp.buffer.buffer)
        r['seen'] = exp.buffer.num_seen_examples
        r['count'] = exp.count
    return r


def restore(exp, state, optimizer=True, rng=True):
    # Install dynamic buffer shapes before strict model load.
    for name, value in state.get('buffers', {}).items():
        parent, _, attr = name.rpartition('.')
        mod = exp.model.get_submodule(parent) if parent else exp.model
        if attr in mod._buffers:
            mod._buffers[attr] = value.to(exp.device)
    exp.model.load_state_dict(state['model'], strict=True)
    if optimizer:
        exp.model_optim.load_state_dict(state['optimizer'])
    for name, p in exp.model.named_parameters():
        p.requires_grad_(state.get('requires_grad', {}).get(name, True))
    for name, value in state.get('flags', {}).items():
        setattr(exp.model, name, value)
    if 'replay' in state:
        exp.buffer.buffer = [x.to(exp.device) for x in state['replay']]
        exp.buffer.num_seen_examples = state['seen']; exp.count = state['count']
    if rng:
        random.setstate(state['rng_python']); np.random.set_state(state['rng_numpy'])
        torch.set_rng_state(state['rng_torch'])
        if state['rng_cuda']:
            torch.cuda.set_rng_state_all(state['rng_cuda'])


def state_hash(exp):
    import hashlib
    h = hashlib.sha256()
    for name, v in sorted(exp.model.state_dict().items()):
        # Only persistent weights and running statistics; per-window RevIN cache excluded.
        h.update(name.encode()); h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def complete_state_hash(state):
    import hashlib
    h=hashlib.sha256()
    def visit(x):
        if torch.is_tensor(x):x=x.detach().cpu().numpy()
        if isinstance(x,np.ndarray):
            h.update(str(x.dtype).encode());h.update(str(x.shape).encode());h.update(x.tobytes())
        elif isinstance(x,dict):
            for k in sorted(x,key=str):h.update(str(k).encode());visit(x[k])
        elif isinstance(x,(list,tuple)):
            h.update(str(len(x)).encode())
            for v in x:visit(v)
        else:h.update(repr(x).encode())
    visit(state)
    return h.hexdigest()


def update(exp, values, targets, t, W, H, method, warmup, audit):
    """At time t, no supervised target later than t is constructed."""
    exp.model.train()
    u = t-H
    if u < W-1:
        return
    origins = [u]
    if method == 'PROCEED' and warmup:
        recent = u-H
        if recent < W-1:
            return
        m = exp.model
        m.freeze_bias(False); m.freeze_adapter(True)
        exp._update_online(batch(values, targets, [recent], W, H, exp.device), mse, exp.model_optim, flag_current=False)
        m.freeze_bias(True); m.backbone.requires_grad_(False); m.freeze_adapter(False)
        exp._update_online(batch(values, targets, [u], W, H, exp.device), mse, exp.model_optim, flag_current=True)
        m.backbone.requires_grad_(True); m.freeze_bias(False); m.freeze_adapter(True)
        origins.insert(0, recent)
    else:
        if method == 'PROCEED':
            exp.model.freeze_adapter(True)
        exp._update_online(batch(values, targets, [u], W, H, exp.device), mse, exp.model_optim)
    audit['updates'] += len(origins)
    lo = min(origins)-W+1
    audit['earliest'] = lo if audit['earliest'] is None else min(audit['earliest'], lo)
    audit['last_target'] = max(o+H for o in origins)
    assert audit['last_target'] <= t
    return origins
