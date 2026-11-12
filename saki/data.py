"""Manifest identity, portable audio paths, and deterministic audio cropping."""
import json
import os
import random
from pathlib import Path
import numpy as np

DATASETS = {
    '01_ESC50': ('ESC-50', 'single-label', 2000, 50),
    '02_UrbanSound8K': ('UrbanSound8K', 'single-label', 8732, 10),
    '03_FSD50K': ('FSD50K', 'multilabel', 10231, 200),
    '04_DCASE17_T4': ('DCASE17-T4', 'multilabel', 416, 17),
    '05_AudioSet': ('AudioSet', 'multilabel', 17233, 527),
    '06_TUT2017': ('TUT2017', 'single-label', 6300, 15),
}


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def seed_all(value):
    import torch
    random.seed(int(value))
    np.random.seed(int(value) % (2**32 - 1))
    torch.manual_seed(int(value))
    torch.cuda.manual_seed_all(int(value))


def configure(threads=2, cpus=None):
    import torch
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    if cpus is not None:
        import psutil
        psutil.Process().cpu_affinity(cpus)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    seed_all(42)


def resolve_audio(row, audio_root):
    prefix = '${DATA_ROOT}/'
    stored = row['audio_path']
    if not stored.startswith(prefix):
        raise ValueError("Unexpected manifest path format.")
    root = Path(audio_root).resolve()
    result = (root / stored[len(prefix):]).resolve()
    if not result.is_relative_to(root):
        raise ValueError("Audio path escapes the configured data directory.")
    return result


def load_manifest(path, classes=None):
    rows = read_json(path)
    if not rows or len({r['sample_index'] for r in rows}) != len(rows):
        raise ValueError("Manifest must contain unique sample indices.")
    if len({r['audio_path'] for r in rows}) != len(rows):
        raise ValueError("Manifest contains duplicate audio paths.")
    for row in rows:
        truth = row['true_indices']
        if not truth or len(set(truth)) != len(truth):
            raise ValueError("Missing or duplicate ground-truth indices.")
        if classes is not None and any(not 0 <= int(i) < len(classes) for i in truth):
            raise ValueError("Ground-truth index outside the candidate class list.")
        if 'seed' not in row:
            raise ValueError("The stored crop seed is required.")
    return rows
