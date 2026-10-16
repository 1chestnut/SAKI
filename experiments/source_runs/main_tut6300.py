from saki.paths import release_path
"""Task25 final independent 2^3 ablation at alpha=.3, Nr=5."""
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
TASK = ROOT / '28_TUT2017_full6300_rerun' / 'main'
AAKV = ROOT / '20A_topP5_aligned_selector_exploration/aakv_all.json'
METHODS = ['CLAP', 'I_iKnow', 'T_AAKV', 'F_Fusion', 'S_Selector', 'TF', 'TS', 'FS', 'TFS_Final']

def seed(v):
    random.seed(v)
    np.random.seed(v % (2 ** 32 - 1))
    torch.manual_seed(v)
    torch.cuda.manual_seed_all(v)

def mod(p):
    s = importlib.util.spec_from_file_location('rt25', str(p))
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m

def nlse(v):
    return (torch.logsumexp(100.0 * v, 0) - math.log(v.numel())) / 100.0

def joint(b, e):
    return b if not e.numel() else nlse(torch.cat((b.reshape(1), e.reshape(-1))))

def hier(b, e):
    return b if not e.numel() else 0.3 * b + 0.7 * nlse(e.reshape(-1))

def rank(z, t):
    o = torch.argsort(z, descending=True).cpu().numpy()
    return min((int(np.where(o == int(y))[0][0]) + 1 for y in t))

def metric(r):
    x = np.asarray(r, float)
    return {'Hit@1': float(100 * np.mean(x <= 1)), 'Hit@3': float(100 * np.mean(x <= 3)), 'Hit@5': float(100 * np.mean(x <= 5)), 'MRR': float(100 * np.mean(1 / x))}

def save(p, x):
    p.write_text(json.dumps(x, ensure_ascii=False, indent=2), encoding='utf-8')

def relation_order(spaces, relations):
    pred = {}
    margin = {}
    for r in relations:
        v = torch.topk(spaces[r], k=min(2, spaces[r].numel())).values
        pred[r] = int(torch.argmax(spaces[r]))
        margin[r] = float(v[0] - v[1]) if v.numel() > 1 else 0.0
    votes = Counter(pred.values())
    mv = max(votes.values())
    cand = sorted((c for c, n in votes.items() if n == mv))
    cc = max(cand, key=lambda c: (sum((margin[r] for r in relations if pred[r] == c)), -c))
    mo = sorted(relations, key=lambda r: (-margin[r], r))
    support = sorted((r for r in relations if pred[r] == cc), key=lambda r: (-margin[r], r))
    return (support + [r for r in mo if r not in support], cc, mv)

def apply_selected(base, top, per_class, chosen, text_mode, fusion):
    """Construct from immutable base; never consumes another method's scores."""
    out = base.clone()
    for ci in top:
        merged = OrderedDict()
        for r in chosen:
            for tail, direct, aakv in per_class[ci][r]:
                merged.setdefault(tail, direct if text_mode == 'direct' else aakv)
        vals = list(merged.values())
        ev = torch.stack(vals) if vals else torch.empty(0, device=base.device)
        out[ci] = joint(base[ci], ev) if fusion == 'joint' else hier(base[ci], ev)
    return out

def main(ds):
    out = TASK / 'test' / ds
    out.mkdir(parents=True, exist_ok=True)
    src = ROOT / '28_TUT2017_full6300_rerun' / 'cache' / ds
    rt = mod(Path(__file__).parent / 'runtime' / ds / 'runtime.py')
    dev = rt.DEVICE
    seed(42)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    clap = rt.CLAP(version='2023', use_cuda=True)
    clap.clap.eval()
    samples = list(rt.iter_samples(rt.load_dataset()))
    frozen = torch.load(src / 'text_embeddings.pt', map_location='cpu')
    le = frozen['labels'].to(dev)
    fe = frozen['evidence'].to(dev)
    fi = frozen['indices']
    cache = torch.load(ROOT / f'18a_relation_oracle_audit/{ds}/results/static_relation_cache.pt', map_location='cpu')
    relations = cache['relations']
    prompts = {p: i for i, p in enumerate(cache['prompts'])}
    pe = cache['prompt_embeddings'].to(dev)
    texts = json.loads(AAKV.read_text(encoding='utf-8'))['texts']
    keys = []
    seen = set()
    records = {r: {} for r in relations}
    for r in relations:
        for ci, items in cache['tails'][r].items():
            a = []
            for tail, prompt in items:
                key = '\t'.join((prompt.split(', ', 1)[0], r, tail))
                if key not in seen:
                    seen.add(key)
                    keys.append(key)
                a.append((tail, prompt, key))
            records[r][int(ci)] = a
    missing = [k for k in keys if k not in texts]
    if missing:
        raise RuntimeError(f'missing {len(missing)} AAKV texts')
    kid = {k: i for i, k in enumerate(keys)}
    chunks = []
    for i in range(0, len(keys), 128):
        chunks.append(F.normalize(rt.get_safe_text_embeddings(clap, [texts[k] for k in keys[i:i + 128]], dev), dim=-1))
    ae = torch.cat(chunks)
    ranks = {m: [] for m in METHODS}
    rows = []
    score_rows = []
    counts = {m: Counter() for m in ['S_Selector', 'TS', 'FS', 'TFS_Final']}
    for i, s in enumerate(tqdm(samples, desc=ds, mininterval=10)):
        sd = int(hashlib.sha256(('42|' + s['audio_path']).encode()).hexdigest()[:8], 16)
        seed(sd)
        audio = torch.load(src / 'audio_cache' / f'{i:06d}.pt', map_location=dev)
        base = (audio @ le.T).squeeze(0)
        base_guard = base.clone()
        top = torch.argsort(base, descending=True)[:5].tolist()
        asim = (audio @ ae.T).squeeze(0)
        per = {ci: {} for ci in top}
        for ci in top:
            for r in relations:
                vals = []
                for tail, prompt, key in records[r].get(ci, []):
                    vals.append((tail, (audio @ pe[prompts[prompt]].reshape(-1, 1)).reshape(()), asim[kid[key]]))
                per[ci][r] = vals
        vec = {'CLAP': base.clone()}
        for name, tm, fm in [('I_iKnow', 'direct', 'joint'), ('T_AAKV', 'aakv', 'joint'), ('F_Fusion', 'direct', 'hier'), ('TF', 'aakv', 'hier')]:
            z = base.clone()
            for ci in top:
                ev = (audio @ fe[fi[ci][tm]].T).reshape(-1)
                z[ci] = joint(base[ci], ev) if fm == 'joint' else hier(base[ci], ev)
            vec[name] = z
        for name, tm, fm in [('S_Selector', 'direct', 'joint'), ('TS', 'aakv', 'joint'), ('FS', 'direct', 'hier'), ('TFS_Final', 'aakv', 'hier')]:
            singles = {r: apply_selected(base, top, per, [r], tm, fm) for r in relations}
            order, cc, mv = relation_order(singles, relations)
            chosen = order[:5]
            vec[name] = apply_selected(base, top, per, chosen, tm, fm)
            counts[name].update(chosen)
            if name == 'S_Selector':
                selected = {}
            selected[name] = {'relations': chosen, 'consensus_class': cc, 'votes': mv}
        if not torch.equal(base, base_guard):
            raise AssertionError('immutable CLAP base was modified')
        rr = {m: rank(vec[m], s['true_indices']) for m in METHODS}
        for m in METHODS:
            ranks[m].append(rr[m])
        rows.append({'sample_index': i, 'audio_path': s['audio_path'], 'true_indices': [int(x) for x in s['true_indices']], 'seed': sd, 'selected': selected, 'ranks': rr})
        score_rows.append(torch.stack([vec[m] for m in METHODS]).cpu().numpy())
    met = {m: metric(ranks[m]) for m in METHODS}
    base_met = met['I_iKnow']
    np.savez_compressed(out / 'predictions.npz', scores=np.asarray(score_rows), methods=METHODS)
    save(out / 'samples.json', rows)
    save(out / 'metrics.json', met)
    save(out / 'protocol.json', {'role': 'final evaluation', 'dataset_partition': 'TUT2017 development (4680) + evaluation (1620)', 'n_samples': len(samples), 'development_set': 'DCASE17-T4', 'alpha': 0.3, 'Nr': 5, 'K': 5, 'M': 3, 'kappa': 100, 'TopP': 'All/disabled', 'hop': 1, 'seed': 42, 'independent_branches': True, 'labels_used_by_selector': False, 'selection_counts': {m: dict(c) for m, c in counts.items()}})
    lines = ['method,Hit@1,Hit@3,Hit@5,MRR,delta_Hit@1_vs_iKnow,delta_MRR_vs_iKnow']
    for m in METHODS:
        lines.append(f"{m},{met[m]['Hit@1']:.8f},{met[m]['Hit@3']:.8f},{met[m]['Hit@5']:.8f},{met[m]['MRR']:.8f},{met[m]['Hit@1'] - base_met['Hit@1']:.8f},{met[m]['MRR'] - base_met['MRR']:.8f}")
    (out / 'metrics.csv').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    save(out / 'progress.json', {'completed': True, 'n': len(samples)})
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('dataset')
    a = p.parse_args()
    main(a.dataset)
