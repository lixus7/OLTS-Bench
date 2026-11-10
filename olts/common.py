"""Small helpers shared by every stage: JSON files, hashes, timestamps and source identity."""
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
UPSTREAM = ROOT / 'third_party' / 'OnlineTSF'
# The main study used exactly this OnlineTSF commit; the submodule is pinned to it.
ONLINETSF_COMMIT = '0a8646199f6dcbf5346f933508b33c5e35b5708b'


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(2**20), b''):
            h.update(chunk)
    return h.hexdigest()


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, allow_nan=False).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, obj):
    """Atomic JSON write: a crash never leaves a half-written file behind."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    q = p.with_name(p.name + '.tmp.' + str(os.getpid()))
    with q.open('w') as f:
        json.dump(obj, f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())
    os.replace(q, p)


def _git(args, cwd):
    try:
        return subprocess.run(['git'] + args, cwd=cwd, capture_output=True, text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ''


def source_info():
    """Which code produced a result: this repository's commit and the OnlineTSF commit."""
    upstream = _git(['rev-parse', 'HEAD'], UPSTREAM)
    return dict(repo_commit=_git(['rev-parse', 'HEAD'], ROOT) or None,
                repo_dirty=bool(_git(['status', '--porcelain', '--untracked-files=no'], ROOT)),
                onlinetsf_commit=upstream or None,
                onlinetsf_expected=ONLINETSF_COMMIT,
                onlinetsf_matches=upstream == ONLINETSF_COMMIT)
