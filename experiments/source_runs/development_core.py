from saki.paths import release_path
"""Task 23: DCASE development-set grid for the complete TFS method."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
os.environ['PYTHONHASHSEED'] = '42'
import hashlib, importlib.util, json, math, random
from collections import Counter, OrderedDict
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
ROOT = Path(release_path('${RUN_ROOT}'))
OUT = ROOT / '23_DCASE_validation_alpha_Nr'
DATASET = '04_DCASE17_T4'
SOURCE = ROOT / '12_shared-abcd-statistics' / DATASET
AAKV_SOURCE = ROOT / '20A_topP5_aligned_selector_exploration' / 'aakv_all.json'
ALPHAS = (0.3, 0.5, 0.7)
N_RELATIONS = (1, 3)
METHODS = [f'TFS_alpha{a:.1f}_Nr{nr}' for nr in N_RELATIONS for a in ALPHAS]

def seed_all(value):
    random.seed(value)
    np.random.seed(value % (2 ** 32 - 1))
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)

def load_module(path):
    spec = importlib.util.spec_from_file_location('task23_runtime', str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def nlse(values):
    return (torch.logsumexp(100.0 * values, dim=0) - math.log(values.numel())) / 100.0

def hierarchical(base_value, evidence, alpha):
    return base_value if not evidence.numel() else alpha * base_value + (1.0 - alpha) * nlse(evidence)

def rank(scores, truths):
    order = torch.argsort(scores, descending=True).cpu().numpy()
    return min((int(np.where(order == int(y))[0][0]) + 1 for y in truths))

def metrics(ranks):
    x = np.asarray(ranks, dtype=float)
    return {'Hit@1': float(100 * np.mean(x <= 1)), 'Hit@3': float(100 * np.mean(x <= 3)), 'Hit@5': float(100 * np.mean(x <= 5)), 'MRR': float(100 * np.mean(1 / x))}

def consensus_order(single_scores, relations):
    predictions, margins = ({}, {})
    for relation in relations:
        values = torch.topk(single_scores[relation], k=min(2, single_scores[relation].numel())).values
        predictions[relation] = int(torch.argmax(single_scores[relation]))
        margins[relation] = float(values[0] - values[1]) if values.numel() > 1 else 0.0
    votes = Counter(predictions.values())
    max_votes = max(votes.values())
    candidates = sorted((c for c, n in votes.items() if n == max_votes))
    consensus = max(candidates, key=lambda c: (sum((margins[r] for r in relations if predictions[r] == c)), -c))
    margin_order = sorted(relations, key=lambda r: (-margins[r], r))
    support = sorted((r for r in relations if predictions[r] == consensus), key=lambda r: (-margins[r], r))
    return (support + [r for r in margin_order if r not in support], consensus, max_votes)

def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    runtime = load_module(SOURCE / 'frozen_inputs' / 'runtime.py')
    device = runtime.DEVICE
    seed_all(42)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    clap = runtime.CLAP(version='2023', use_cuda=True)
    clap.clap.eval()
    samples = list(runtime.iter_samples(runtime.load_dataset()))
    frozen = torch.load(SOURCE / 'text_embeddings.pt', map_location='cpu')
    label_embeddings = frozen['labels'].to(device)
    cache = torch.load(ROOT / f'18a_relation_oracle_audit/{DATASET}/results/static_relation_cache.pt', map_location='cpu')
    relations = cache['relations']
    texts = json.loads(AAKV_SOURCE.read_text(encoding='utf-8'))['texts']
    keys, seen = ([], set())
    relation_items = {r: {} for r in relations}
    for relation in relations:
        for ci, items in cache['tails'][relation].items():
            records = []
            for tail, prompt in items:
                key = '\t'.join((prompt.split(', ', 1)[0], relation, tail))
                if key not in seen:
                    seen.add(key)
                    keys.append(key)
                records.append((tail, key))
            relation_items[relation][int(ci)] = records
    missing = [k for k in keys if k not in texts]
    if missing:
        raise RuntimeError(f'AAKV cache missing {len(missing)} keys')
    key_index = {k: i for i, k in enumerate(keys)}
    chunks = []
    for start in range(0, len(keys), 128):
        chunks.append(F.normalize(runtime.get_safe_text_embeddings(clap, [texts[k] for k in keys[start:start + 128]], device), dim=-1))
    aakv_embeddings = torch.cat(chunks)
    rank_lists = {m: [] for m in ['CLAP'] + METHODS}
    rows = []
    selection_counts = {m: Counter() for m in METHODS}
    for index, sample in enumerate(tqdm(samples, desc=f'{DATASET}-grid', mininterval=10)):
        sample_seed = int(hashlib.sha256(('42|' + sample['audio_path']).encode()).hexdigest()[:8], 16)
        seed_all(sample_seed)
        audio = torch.load(SOURCE / 'audio_cache' / f'{index:06d}.pt', map_location=device)
        base = (audio @ label_embeddings.T).squeeze(0)
        top_classes = torch.argsort(base, descending=True)[:5].tolist()
        evidence_scores = (audio @ aakv_embeddings.T).squeeze(0)
        per_class = {ci: {} for ci in top_classes}
        for ci in top_classes:
            for relation in relations:
                rec = relation_items[relation].get(ci, [])
                per_class[ci][relation] = [(tail, evidence_scores[key_index[key]]) for tail, key in rec]
        vectors = {'CLAP': base}
        selected_meta = {}
        for alpha in ALPHAS:
            singles = {}
            for relation in relations:
                z = base.clone()
                for ci in top_classes:
                    vals = [score for _, score in per_class[ci][relation]]
                    z[ci] = hierarchical(base[ci], torch.stack(vals) if vals else torch.empty(0, device=device), alpha)
                singles[relation] = z
            ordered, consensus, votes = consensus_order(singles, relations)
            for nr in N_RELATIONS:
                method = f'TFS_alpha{alpha:.1f}_Nr{nr}'
                chosen = ordered[:nr]
                z = base.clone()
                for ci in top_classes:
                    merged = OrderedDict()
                    for relation in chosen:
                        for tail, score in per_class[ci][relation]:
                            merged.setdefault(tail, score)
                    vals = list(merged.values())
                    z[ci] = hierarchical(base[ci], torch.stack(vals) if vals else torch.empty(0, device=device), alpha)
                vectors[method] = z
                selection_counts[method].update(chosen)
                selected_meta[method] = {'relations': chosen, 'consensus_class': consensus, 'votes': votes}
        sample_ranks = {m: rank(vectors[m], sample['true_indices']) for m in vectors}
        for method, value in sample_ranks.items():
            rank_lists[method].append(value)
        rows.append({'sample_index': index, 'audio_path': sample['audio_path'], 'true_indices': [int(x) for x in sample['true_indices']], 'seed': sample_seed, 'selected': selected_meta, 'ranks': sample_ranks})
    result = {m: metrics(rank_lists[m]) for m in rank_lists}
    best_mrr = max((result[m]['MRR'] for m in METHODS))
    eligible = [m for m in METHODS if best_mrr - result[m]['MRR'] < 0.1]

    def simplicity(method):
        nr = int(method.rsplit('Nr', 1)[1])
        alpha = float(method.split('alpha', 1)[1].split('_', 1)[0])
        return (nr, abs(alpha - 0.5), alpha)
    selected = min(eligible, key=simplicity)
    save(OUT / 'metrics.json', result)
    save(OUT / 'samples.json', rows)
    save(OUT / 'selected_config.json', {'primary_metric': 'MRR', 'tie_tolerance_pp': 0.1, 'best_MRR': best_mrr, 'eligible_within_tolerance': eligible, 'selected': selected, 'selection_rule': 'highest MRR; within 0.1 pp prefer Nr=1, then alpha closest to 0.5'})
    save(OUT / 'protocol.json', {'role': 'development/validation only' if DATASET == '04_DCASE17_T4' else 'test-set diagnostic only', 'dataset': DATASET, 'n': len(samples), 'complete_method': 'TFS (AAKV + hierarchical fusion + Consensus-Margin selector)', 'grid': {'alpha': ALPHAS, 'Nr': N_RELATIONS}, 'fixed': {'hop': 1, 'K': 5, 'M': 3, 'TopP': 'All/disabled', 'kappa': 100, 'seed': 42, 'AAKV': 'frozen', 'relation_pool': 'fixed 47 relations'}, 'ground_truth_use': 'only aggregate validation metrics and configuration selection; never per-sample routing', 'selection_counts': {m: dict(c) for m, c in selection_counts.items()}})
    header = 'method,alpha,Nr,Hit@1,Hit@3,Hit@5,MRR,selected\n'
    lines = []
    for m in METHODS:
        a = float(m.split('alpha', 1)[1].split('_', 1)[0])
        nr = int(m.rsplit('Nr', 1)[1])
        v = result[m]
        lines.append(f"{m},{a:.1f},{nr},{v['Hit@1']:.8f},{v['Hit@3']:.8f},{v['Hit@5']:.8f},{v['MRR']:.8f},{str(m == selected).lower()}")
    (OUT / 'validation_grid.csv').write_text(header + '\n'.join(lines) + '\n', encoding='utf-8')
    save(OUT / 'progress.json', {'completed': True, 'n': len(samples)})
    print(json.dumps({'metrics': result, 'selected': selected}, indent=2))
if __name__ == '__main__':
    main()
