"""One immutable job: resource profile, offline fit, or causal stream (the main study's worker).

Run as: python -m olts.worker --run RUN_DIR --job JOB_ID   (reads RUN_DIR/configs/JOB_ID.json)
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import gc
import math
import resource
import time
import traceback
from pathlib import Path
import numpy as np
import torch
from .common import read, write, sha, now, digest, source_info
from .data import load
from .engine import arguments, make, seed_all, batch, mse, prediction, snapshot, restore, state_hash, complete_state_hash, update


def sync():
    if torch.cuda.is_available(): torch.cuda.synchronize()


def save_state(path, obj):
    tmp = str(path)+'.tmp'; torch.save(obj,tmp); os.replace(tmp,path)


def fill_replay(exp, values, targets, W, H):
    # Upstream Buffer.add_data is only valid for the native batch-one online API.
    for k in range(500):
        bb=batch(values,targets,[W-1+k],W,H,exp.device)
        exp.buffer.add_data(*(bb+(torch.tensor([k],device=exp.device),)))
    assert exp.buffer.num_seen_examples==500
    exp.count=500


def profile_one(c, values, targets, meta, method):
    W,H=c['W'],c['H']; nt=meta['boundaries'][1]
    seed_all(c['seed']); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    args=arguments(c['model'],c['dataset'],W,H,values.shape[1],1e-4,'cuda')
    exp=make(args,method); exp.model.train()
    origins=np.arange(W-1,W-1+32)
    b=batch(values,targets,origins,W,H,exp.device)
    steps=[]
    for j in range(10):
        sync(); start=time.perf_counter(); loss,_=exp._update(b,mse,exp.model_optim);sync()
        if not torch.isfinite(loss):raise FloatingPointError('Profile loss')
        steps.append(time.perf_counter()-start)
    if method=='ER':
        # Include a fully occupied native reservoir and replay gradients in the peak.
        fill_replay(exp, values, targets, W, H)
    audit=dict(updates=0,earliest=None,last_target=None)
    online=[]
    for t in range(max(W+2*H,1000),max(W+2*H,1000)+10):
        assert t < nt
        sync(); start=time.perf_counter()
        update(exp,values,targets,t,W,H,method,True,audit)
        prediction(exp,values,t,W);sync();online.append(time.perf_counter()-start)
    total=torch.cuda.get_device_properties(0).total_memory
    peak=torch.cuda.max_memory_reserved()
    return dict(method=method,peak_reserved_bytes=peak,total_bytes=total,
        parameters=sum(p.numel() for p in exp.model.parameters()),
        reserve_fraction=1-peak/total,step_seconds=float(np.median(steps[2:])),
        online_seconds=float(np.median(online[2:])), status='PASS' if peak<=.8*total else 'RESOURCE_LIMIT')

def profile(c, out, values, targets, meta):
    stats=[]
    for method in ('Naive','ER','PROCEED') if c['W'] in (96,512,1024) else ('Naive',):
        try:
            stat=profile_one(c,values,targets,meta,method)
        except torch.cuda.OutOfMemoryError as exc:
            stat=dict(method=method,status='RESOURCE_LIMIT',reason='OOM',message=str(exc))
        stats.append(stat)
        write(out/'PROFILE_PROGRESS.json',stats)
        gc.collect();torch.cuda.empty_cache()
    return dict(status='PASS' if stats[0]['status']=='PASS' else 'RESOURCE_LIMIT',
        profiles=stats,batch_size=32,precision='FP32',workers_per_gpu=1,
        cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),gpu=torch.cuda.get_device_name(0),
        cpu_maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def validation(exp, values, targets, W, H, start, end):
    exp.model.eval();sse=0.;count=0
    with torch.no_grad():
        for t in range(start-1,end-H,32):
            origins=list(range(t,min(t+32,end-H)))
            b=batch(values,targets,origins,W,H,exp.device)
            pred=exp.forward(b);mask=torch.isfinite(b[1])
            if not torch.isfinite(pred).all():raise FloatingPointError('Nonfinite VALIDATION')
            sse+=float((pred[mask]-b[1][mask]).double().square().sum());count+=int(mask.sum())
    if not count:raise ValueError('No observed VALIDATION targets')
    return sse/count


def fit(c, out, values, targets, meta, run):
    W,H=c['W'],c['H'];nt,nv=meta['boundaries'][1:3]
    seed_all(c['seed']); args=arguments(c['model'],c['dataset'],W,H,values.shape[1],c['lr'],'cuda')
    method=c.get('architecture','Naive')
    if method=='PROCEED':
        parent=Path(run)/'jobs'/c['initial_job']
        assert read(parent/'DONE.json')['status']=='PASS'
        initial=torch.load(parent/'checkpoint.pt',map_location='cpu',weights_only=False)
        init_path=out/'backbone_initial.pt'
        save_state(init_path,dict(model=initial['state']['model'],model_optim=initial['state']['optimizer']))
        args.load_path=str(init_path)
    exp=make(args,method)
    for g in exp.model_optim.param_groups:g['lr']=c['lr']
    if method=='PROCEED':
        # Native run.py loads the pretrained backbone, initializes adapters, then
        # learns them in update_valid. Do not invent an extra 50-epoch backbone fit.
        save_state(out/'checkpoint.pt',dict(state=snapshot(exp),args=vars(args),config=c,epoch=0,
            steps=0,VALIDATION_MSE=None,data_sha256=meta['arrays_sha256'],
            source=source_info()))
        return dict(status='PASS',evaluation='INITIALIZATION_ONLY',VALIDATION_MSE=None,
            checkpoint_sha256=sha(out/'checkpoint.pt'),chosen_epoch=0,epochs=0,total_optimizer_steps=0,
            train_seconds=0,initial_job=c['initial_job'],configuration=c,
            note='Native backbone-load and adapter initialization; adapter fitting is counted in causal VALIDATION warmup')
    origins=np.arange(c['common_W']-1,nt-H,dtype=np.int64)
    assert len(origins)>=1024 and W<=c['common_W']
    order_rng=np.random.default_rng(c['seed'])
    epoch_times=[];best=float('inf');bad=0;steps=0;history=[]
    # 50 epochs as in the main study; max_epochs exists only for quick smoke tests.
    for epoch in range(1,int(c.get('max_epochs',50))+1):
        sync();start=time.perf_counter();exp.model.train()
        order=order_rng.permutation(origins)
        for k in range(0,len(order),32):
            b=batch(values,targets,order[k:k+32],W,H,exp.device)
            loss,_=exp._update(b,mse,exp.model_optim)
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite TRAIN')
            steps+=1
        score=validation(exp,values,targets,W,H,nt,nv)
        sync();elapsed=time.perf_counter()-start;epoch_times.append(elapsed)
        record=dict(epoch=epoch,steps=steps,VALIDATION_MSE=score,seconds=elapsed)
        history.append(record);write(out/'PROGRESS.json',dict(time=now(),**record))
        write(out/'history.json',history)
        improved=score<best
        if improved:best=score;bad=0
        else:bad+=1
        # C keeps final fixed-step checkpoint; T keeps best and may stop at patience 5.
        if c['track']=='C' or improved:
            save_state(out/'checkpoint.pt',dict(state=snapshot(exp),args=vars(args),config=c,epoch=epoch,
                steps=steps,VALIDATION_MSE=score,order_rng=order_rng.bit_generator.state,
                data_sha256=meta['arrays_sha256'],source=source_info()))
        if c['track']=='T' and bad>=5:break
    chosen=torch.load(out/'checkpoint.pt',map_location='cpu',weights_only=False)
    return dict(status='PASS',evaluation='VALIDATION',VALIDATION_MSE=chosen['VALIDATION_MSE'],
        checkpoint_sha256=sha(out/'checkpoint.pt'),chosen_epoch=chosen['epoch'],epochs=epoch,
        total_optimizer_steps=steps,common_origins=len(origins),common_origin_hash=digest(origins.tolist()),
        train_seconds=sum(epoch_times),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        initial_job=c.get('initial_job'),configuration=c)


def stream(c, out, values, targets, mu, scale, meta, run):
    W,H=c['W'],c['H']; nt,ve,te=meta['boundaries'][1:]
    method=c['method'];parent=Path(run)/'jobs'/c['initial_job']
    assert read(parent/'DONE.json')['status']=='PASS'
    parent_sha=sha(parent/'checkpoint.pt')
    state=torch.load(parent/'checkpoint.pt',map_location='cpu',weights_only=False)
    seed_all(c['seed']); args=arguments(c['model'],c['dataset'],W,H,values.shape[1],c['lr'],'cuda')
    architecture='PROCEED' if method.startswith('PROCEED') else ('ER' if method.startswith('ER') else 'Naive')
    # Native Proceed saved two optimizer groups when built from an offline backbone.
    if architecture=='PROCEED':
        args.load_path=str(parent/'backbone_initial.pt')
    exp=make(args,architecture)
    restore(exp,state['state'],rng=True)
    for g in exp.model_optim.param_groups:g['lr']=c['online_lr']
    freeze=method=='Frozen' or method.endswith('-frozen')
    evaluate=c['evaluation']
    start,end=(nt,ve) if evaluate=='VALIDATION' else (ve,te)
    if evaluate=='VALIDATION':
        # Predeclared first half supplies warm-up; rank candidates on the second half.
        score_start=nt+(ve-nt)//2
    else:score_start=ve
    audit=dict(updates=0,earliest=None,last_target=None)
    records=[];update_seconds=[];predict_seconds=[];response_seconds=[]
    stage_updates={'VALIDATION_WARMUP':[],'TEST':[],'VALIDATION_SCORING':[]}
    n=end-score_start-H+1
    assert n>0
    predfile=np.lib.format.open_memmap(out/'predictions.npy',mode='w+',dtype='float32',shape=(n,H,values.shape[1]))
    predhash=__import__('hashlib').sha256()
    first_hash=state_hash(exp);test_start_hash=None;test_start_complete_hash=None
    # A paired PROCEED-frozen branch follows identical pre-TEST warmup before freezing.
    warm_method=architecture
    for t in range(nt-1,end):
        # Score only immutable predictions whose entire target is now observed.
        u=t-H
        if score_start-1<=u<end-H:
            ix=u-(score_start-1);p=np.array(predfile[ix],dtype='float64')
            y=targets[u+1:u+H+1].astype('float64');mask=np.isfinite(y)
            err=np.where(mask,p-y,0);count=int(mask.sum())
            records[ix].update(count=count,sse=float(np.square(err).sum()),sae=float(np.abs(err).sum()),
                raw_sse=float(np.square(err*scale).sum()),raw_sae=float(np.abs(err*scale).sum()),scored_at=t)
        if t>=end-H:continue # No forecast outside evaluation segment; no unnecessary tail updates.
        sync();begin=time.perf_counter()
        if architecture=='PROCEED' and method!='Frozen' and nt+H-1<=t<=ve-1:
            # Defer native (recent,current) validation pairs until current's full
            # label arrives. Last legal VAL pair completes before TEST-state clone.
            update(exp,values,targets,t,W,H,'PROCEED',True,audit)
        if t==ve-1:
            test_start_hash=state_hash(exp)
            before_test=snapshot(exp);test_start_complete_hash=complete_state_hash(before_test)
            save_state(out/'deployment_start.pt',dict(state=before_test,audit=audit.copy(),initial_sha=parent_sha))
        do_update=(not freeze) or (method.endswith('-frozen') and t<ve-1)
        if architecture=='PROCEED' and t<ve-1:do_update=False
        if do_update:
            update(exp,values,targets,t,W,H,warm_method,False,audit)
        sync();after_update=time.perf_counter();update_seconds.append(after_update-begin)
        stage='TEST' if evaluate=='TEST' and t>=ve-1 else ('VALIDATION_SCORING' if evaluate=='VALIDATION' and t>=score_start-1 else 'VALIDATION_WARMUP')
        stage_updates[stage].append(after_update-begin)
        # Warmup still runs before scoring; no resetting at score_start.
        if t<score_start-1:continue
        p=prediction(exp,values,t,W);sync();after_predict=time.perf_counter()
        ix=t-(score_start-1);predfile[ix]=p;predhash.update(p.tobytes())
        earliest=audit['earliest']
        K=max(W,t-earliest+1) if earliest is not None else W
        records.append(dict(origin=t,H=H,W=W,earliest_deployment_observation=earliest,history_span=K,
            updates_so_far=audit['updates'],last_update_target=audit['last_target'],prediction_sha256=__import__('hashlib').sha256(p.tobytes()).hexdigest()))
        predict_seconds.append(after_predict-after_update);response_seconds.append(after_predict-begin)
        if ix%500==0:
            predfile.flush();write(out/'PROGRESS.json',dict(time=now(),stage=evaluate,origin=t,total=n,completed=ix+1,updates=audit['updates']))
    predfile.flush()
    assert len(records)==n and all('sse' in r for r in records)
    frozen_reference=test_start_hash if method.endswith('-frozen') else first_hash
    if freeze and state_hash(exp)!=frozen_reference:raise AssertionError('Frozen persistent state changed')
    write(out/'per_origin.json',records)
    save_state(out/'deployment_end.pt',dict(state=snapshot(exp),audit=audit))
    total=sum(r['count'] for r in records)
    def timing(v):return dict(count=len(v),total_seconds=float(sum(v)),p50=float(np.median(v)) if v else None,p95=float(np.quantile(v,.95)) if v else None)
    return dict(status='PASS',evaluation=evaluate,method=method,origins=n,observed_targets=total,
        MSE=sum(r['sse'] for r in records)/total,MAE=sum(r['sae'] for r in records)/total,
        raw_MSE=sum(r['raw_sse'] for r in records)/total,raw_MAE=sum(r['raw_sae'] for r in records)/total,
        parent_checkpoint_sha256=parent_sha,initial_parameter_hash=first_hash,TEST_start_hash=test_start_hash,
        TEST_start_complete_hash=test_start_complete_hash,
        prediction_content_sha256=predhash.hexdigest(),files={p.name:sha(p) for p in out.iterdir() if p.suffix in ('.npy','.pt') or p.name=='per_origin.json'},
        audit=audit,update_cost=timing(update_seconds),prediction_cost=timing(predict_seconds),response_cost=timing(response_seconds),
        update_cost_by_stage={k:timing(v) for k,v in stage_updates.items()},
        parameters=sum(p.numel() for p in exp.model.parameters()),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(),configuration=c)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);ap.add_argument('--job',required=True);a=ap.parse_args()
    root=Path(a.run);out=root/'jobs'/a.job
    source=source_info();c=read(root/'configs'/(a.job+'.json'))
    if out.exists():raise FileExistsError('Refuse to overwrite prior job: '+str(out))
    out.mkdir(parents=True);write(out/'START.json',dict(time=now(),pid=os.getpid(),cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),config=c,source=source))
    start=time.perf_counter()
    try:
        if not torch.cuda.is_available():raise RuntimeError('GPU allocation required')
        torch.set_num_threads(4)
        values,targets,mu,scale,meta=load(root,c['dataset'],c['split'])
        if sha(root/'data'/(c['dataset']+'_'+c['split'])/'arrays.npz')!=meta['arrays_sha256']:raise RuntimeError('Data changed')
        torch.cuda.reset_peak_memory_stats()
        if c['kind']=='profile':result=profile(c,out,values,targets,meta)
        elif c['kind']=='fit':result=fit(c,out,values,targets,meta,root)
        elif c['kind']=='stream':result=stream(c,out,values,targets,mu,scale,meta,root)
        else:raise ValueError(c['kind'])
        result.update(finished=now(),wall_seconds=time.perf_counter()-start,source=source,job=a.job)
        write(out/'DONE.json',result)
    except BaseException as exc:
        write(out/'FAILED.json',dict(time=now(),type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc(),config=c))
        raise


if __name__=='__main__':main()
