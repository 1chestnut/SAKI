from saki.paths import release_path
"""Task26: controlled AAKV relation-selection comparison."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
os.environ['PYTHONHASHSEED'] = '42'
import argparse, hashlib, importlib.util, json, math, random
from collections import Counter, OrderedDict
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
ROOT = Path(release_path('${RUN_ROOT}'))
TASK = ROOT / '28_TUT2017_full6300_rerun' / 'selector'
AAKV = ROOT / '20A_topP5_aligned_selector_exploration/aakv_all.json'
METHODS = ['AAKV-FrozenRq', 'AAKV-Frequency5', 'AAKV-Random5-s42', 'AAKV-Random5-s43', 'AAKV-Random5-s44', 'AAKV-Selector5']

def seed_all(v):
    random.seed(v)
    np.random.seed(v % (2 ** 32 - 1))
    torch.manual_seed(v)
    torch.cuda.manual_seed_all(v)

def load_module(path):
    spec = importlib.util.spec_from_file_location('task26_runtime', str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def nlse(v):
    return (torch.logsumexp(100.0 * v, 0) - math.log(v.numel())) / 100.0

def fuse(base, evidence):
    return base if not evidence.numel() else 0.3 * base + 0.7 * nlse(evidence.reshape(-1))

def apply_relations(base, top, evidence, relations):
    out = base.clone()
    for ci in top:
        merged = OrderedDict()
        for rel in relations:
            for tail, score in evidence[ci].get(rel, []):
                merged.setdefault(tail, score)
        vals = torch.stack(list(merged.values())) if merged else torch.empty(0, device=base.device)
        out[ci] = fuse(base[ci], vals)
    return out

def selector_order(spaces, relations):
    pred, margin = ({}, {})
    for rel in relations:
        vals = torch.topk(spaces[rel], k=min(2, spaces[rel].numel())).values
        pred[rel] = int(torch.argmax(spaces[rel]))
        margin[rel] = float(vals[0] - vals[1]) if vals.numel() > 1 else 0.0
    votes = Counter(pred.values())
    mv = max(votes.values())
    candidates = sorted((c for c, n in votes.items() if n == mv))
    cc = max(candidates, key=lambda c: (sum((margin[r] for r in relations if pred[r] == c)), -c))
    margin_order = sorted(relations, key=lambda r: (-margin[r], r))
    support = sorted((r for r in relations if pred[r] == cc), key=lambda r: (-margin[r], r))
    return (support + [r for r in margin_order if r not in support], cc, mv)

def rank(scores, truth):
    order = torch.argsort(scores, descending=True).cpu().numpy()
    return min((int(np.where(order == int(y))[0][0]) + 1 for y in truth))

def metrics(ranks):
    x = np.asarray(ranks, float)
    return {'Hit@1': float(100 * np.mean(x <= 1)), 'Hit@3': float(100 * np.mean(x <= 3)), 'Hit@5': float(100 * np.mean(x <= 5)), 'MRR': float(100 * np.mean(1 / x))}

def main(ds):
    out = TASK / ds
    out.mkdir(parents=True, exist_ok=True)
    source = ROOT / '28_TUT2017_full6300_rerun' / 'cache' / ds
    runtime = load_module(Path(__file__).parent / 'runtime' / ds / 'runtime.py')
    device = runtime.DEVICE
    config = json.loads((ROOT / '05b_strict-map-k5m3-freeze' / ds / 'config.json').read_text())
    seed_all(42)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    clap = runtime.CLAP(version='2023', use_cuda=True)
    clap.clap.eval()
    samples = list(runtime.iter_samples(runtime.load_dataset()))
    labels = torch.load(source / 'text_embeddings.pt', map_location='cpu')['labels'].to(device)
    cache = torch.load(ROOT / '18a_relation_oracle_audit' / ds / 'results/static_relation_cache.pt', map_location='cpu')
    relations = list(cache['relations'])
    text_map = json.loads(AAKV.read_text(encoding='utf-8'))['texts']
    frozen = [r for r in config['relations'] if r in relations]
    frequency = {r: sum((len(v) for v in cache['tails'][r].values())) for r in relations}
    frequency5 = sorted(relations, key=lambda r: (-frequency[r], r))[:5]
    keys, seen, records = ([], set(), {r: {} for r in relations})
    for rel in relations:
        for ci, items in cache['tails'][rel].items():
            row = []
            for tail, prompt in items:
                key = '\t'.join((prompt.split(', ', 1)[0], rel, tail))
                if key not in seen:
                    seen.add(key)
                    keys.append(key)
                row.append((tail, key))
            records[rel][int(ci)] = row
    missing = [k for k in keys if k not in text_map]
    if missing:
        raise RuntimeError(f'missing {len(missing)} AAKV texts')
    key_id = {k: i for i, k in enumerate(keys)}
    chunks = []
    for i in range(0, len(keys), 128):
        chunks.append(F.normalize(runtime.get_safe_text_embeddings(clap, [text_map[k] for k in keys[i:i + 128]], device), dim=-1))
    embeddings = torch.cat(chunks)
    ranks = {m: [] for m in METHODS}
    rows = []
    counts = {m: Counter() for m in METHODS}
    for i, sample in enumerate(tqdm(samples, desc=ds, mininterval=10)):
        sample_seed = int(hashlib.sha256(('42|' + sample['audio_path']).encode()).hexdigest()[:8], 16)
        seed_all(sample_seed)
        audio = torch.load(source / 'audio_cache' / f'{i:06d}.pt', map_location=device)
        base = (audio @ labels.T).squeeze(0)
        top = torch.argsort(base, descending=True)[:5].tolist()
        sim = (audio @ embeddings.T).squeeze(0)
        evidence = {ci: {} for ci in top}
        for ci in top:
            for rel in relations:
                evidence[ci][rel] = [(tail, sim[key_id[key]]) for tail, key in records[rel].get(ci, [])]
        spaces = {rel: apply_relations(base, top, evidence, [rel]) for rel in relations}
        order, cc, mv = selector_order(spaces, relations)
        selected = order[:5]
        choices = {'AAKV-FrozenRq': frozen, 'AAKV-Frequency5': frequency5, 'AAKV-Selector5': selected}
        for rs in (42, 43, 44):
            rng = random.Random(int(hashlib.sha256(f"{rs}|{sample['audio_path']}".encode()).hexdigest()[:16], 16))
            choices[f'AAKV-Random5-s{rs}'] = sorted(rng.sample(relations, min(5, len(relations))))
        rr = {}
        for method in METHODS:
            vec = apply_relations(base, top, evidence, choices[method])
            rr[method] = rank(vec, sample['true_indices'])
            ranks[method].append(rr[method])
            counts[method].update(choices[method])
        rows.append({'sample_index': i, 'audio_path': sample['audio_path'], 'true_indices': [int(x) for x in sample['true_indices']], 'seed': sample_seed, 'relations': choices, 'selector_consensus_class': cc, 'selector_votes': mv, 'ranks': rr})
    result = {m: metrics(ranks[m]) for m in METHODS}
    (out / 'samples.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding='utf-8')
    (out / 'metrics.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    lines = ['method,Hit@1,Hit@3,Hit@5,MRR'] + [f"{m},{result[m]['Hit@1']:.8f},{result[m]['Hit@3']:.8f},{result[m]['Hit@5']:.8f},{result[m]['MRR']:.8f}" for m in METHODS]
    (out / 'metrics.csv').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    protocol = {'only_changed_variable': 'relation selection', 'text': 'AAKV for every method', 'fusion': 'hierarchical NormLSE', 'alpha': 0.3, 'Nr': 5, 'K': 5, 'M': 3, 'TopP': 'All/disabled', 'hop': 1, 'seed': 42, 'labels_used': False, 'frozen_relations': frozen, 'frequency_definition': 'number of cached class-tail evidence items; no audio labels', 'frequency5': frequency5, 'random_seeds': [42, 43, 44], 'dataset_partition': 'TUT2017 development (4680) + evaluation (1620)', 'n_samples': len(samples), 'selection_counts': {m: dict(c) for m, c in counts.items()}}
    (out / 'protocol.json').write_text(json.dumps(protocol, ensure_ascii=False, indent=2), encoding='utf-8')
    (out / 'progress.json').write_text(json.dumps({'completed': True, 'n': len(samples)}, indent=2), encoding='utf-8')
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('dataset')
    main(p.parse_args().dataset)
