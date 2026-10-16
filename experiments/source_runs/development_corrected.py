from saki.paths import release_path
"""Revalidate alpha and Nr with complete class labels and corrected AAKV text."""
import csv
import hashlib
import importlib.util
import json
from collections import Counter, OrderedDict
from pathlib import Path
import torch
import torch.nn.functional as F
from tqdm import tqdm
ROOT = Path(release_path('${RUN_ROOT}'))
OUT = ROOT / '31_corrected_aakv_cache/dcase_validation'
SOURCE = ROOT / '12_shared-abcd-statistics/04_DCASE17_T4'
spec = importlib.util.spec_from_file_location('validation_core', ROOT / 'task23_dcase_validation.py')
core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)
ALPHAS = (0.3, 0.5, 0.7)
N_RELATIONS = (1, 3, 5, 10, 47)
METHODS = [f'TFS_alpha{a:.1f}_Nr{nr}' for nr in N_RELATIONS for a in ALPHAS]

def main():
    OUT.mkdir(exist_ok=True)
    runtime = core.load_module(SOURCE / 'frozen_inputs/runtime.py')
    device = runtime.DEVICE
    core.seed_all(42)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    clap = runtime.CLAP(version='2023', use_cuda=True)
    clap.clap.eval()
    samples = list(runtime.iter_samples(runtime.load_dataset()))
    assert len(samples) == 419
    frozen = torch.load(SOURCE / 'text_embeddings.pt', map_location='cpu')
    labels_emb = frozen['labels'].to(device)
    cache = torch.load(ROOT / '18a_relation_oracle_audit/04_DCASE17_T4/results/static_relation_cache.pt', map_location='cpu')
    labels = cache['labels']
    relations = cache['relations']
    texts = json.loads((ROOT / '31_corrected_aakv_cache/aakv_dcase.json').read_text(encoding='utf-8'))['texts']
    keys, seen = ([], set())
    relation_items = {r: {} for r in relations}
    for relation in relations:
        for ci, items in cache['tails'][relation].items():
            ci = int(ci)
            head = labels[ci]
            records = []
            for tail, prompt in items:
                assert prompt.startswith(head + ', '), (head, prompt)
                k = '\t'.join((head, relation, tail))
                if k not in seen:
                    seen.add(k)
                    keys.append(k)
                records.append((tail, k))
            relation_items[relation][ci] = records
    missing = [k for k in keys if k not in texts]
    assert not missing, f'Missing {len(missing)} AAKV keys'
    key_index = {k: i for i, k in enumerate(keys)}
    chunks = [F.normalize(runtime.get_safe_text_embeddings(clap, [texts[k] for k in keys[i:i + 128]], device), dim=-1) for i in range(0, len(keys), 128)]
    aakv_embeddings = torch.cat(chunks)
    ranks = {m: [] for m in ['CLAP'] + METHODS}
    rows = []
    for index, sample in enumerate(tqdm(samples, desc='DCASE corrected grid', mininterval=10)):
        sample_seed = int(hashlib.sha256(('42|' + sample['audio_path']).encode()).hexdigest()[:8], 16)
        core.seed_all(sample_seed)
        audio = torch.load(SOURCE / 'audio_cache' / f'{index:06d}.pt', map_location=device)
        base = (audio @ labels_emb.T).squeeze(0)
        top_classes = torch.argsort(base, descending=True)[:5].tolist()
        evidence_scores = (audio @ aakv_embeddings.T).squeeze(0)
        per_class = {ci: {} for ci in top_classes}
        for ci in top_classes:
            for relation in relations:
                per_class[ci][relation] = [(tail, evidence_scores[key_index[k]]) for tail, k in relation_items[relation].get(ci, [])]
        vectors = {'CLAP': base}
        selected_meta = {}
        for alpha in ALPHAS:
            singles = {}
            for relation in relations:
                z = base.clone()
                for ci in top_classes:
                    vals = [score for _, score in per_class[ci][relation]]
                    evidence = torch.stack(vals) if vals else torch.empty(0, device=device)
                    z[ci] = core.hierarchical(base[ci], evidence, alpha)
                singles[relation] = z
            ordered, consensus, votes = core.consensus_order(singles, relations)
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
                    evidence = torch.stack(vals) if vals else torch.empty(0, device=device)
                    z[ci] = core.hierarchical(base[ci], evidence, alpha)
                vectors[method] = z
                selected_meta[method] = {'relations': chosen, 'consensus_class': consensus, 'votes': votes}
        sample_ranks = {m: core.rank(z, sample['true_indices']) for m, z in vectors.items()}
        for method, value in sample_ranks.items():
            ranks[method].append(value)
        rows.append({'sample_index': index, 'audio_path': sample['audio_path'], 'true_indices': [int(x) for x in sample['true_indices']], 'selected': selected_meta, 'ranks': sample_ranks})
    result = {m: core.metrics(v) for m, v in ranks.items()}
    eligible = [m for m in METHODS if not m.endswith('_Nr47')]
    best_hit = max((result[m]['Hit@1'] for m in eligible))
    hit_ties = [m for m in eligible if result[m]['Hit@1'] == best_hit]
    best_mrr = max((result[m]['MRR'] for m in hit_ties))
    mrr_ties = [m for m in hit_ties if result[m]['MRR'] == best_mrr]
    selected = min(mrr_ties, key=lambda m: int(m.rsplit('Nr', 1)[1]))
    decision = {'selected': selected, 'best_Hit@1': best_hit, 'best_MRR_within_Hit@1_ties': best_mrr, 'Hit@1_tied_configs': hit_ties, 'selection_rule': 'Hit@1, then exact-tie MRR, then smaller Nr', 'Nr47_role': 'all-relations diagnostic, not eligible'}
    core.save(OUT / 'metrics.json', result)
    core.save(OUT / 'samples.json', rows)
    core.save(OUT / 'selected_config_hit1_first.json', decision)
    with (OUT / 'validation_grid.csv').open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['method', 'alpha', 'Nr', 'Hit@1', 'Hit@3', 'Hit@5', 'MRR', 'selected'])
        for m in METHODS:
            w.writerow([m, m.split('alpha', 1)[1].split('_', 1)[0], m.rsplit('Nr', 1)[1], *(result[m][k] for k in ('Hit@1', 'Hit@3', 'Hit@5', 'MRR')), m == selected])
    old = json.loads((ROOT / '23C_DCASE_validation_expanded_Nr/metrics.json').read_text(encoding='utf-8'))
    core.save(OUT / 'comparison_vs_old.json', {m: {k: result[m][k] - old[m][k] for k in result[m]} for m in result if m in old})
    core.save(OUT / 'progress.json', {'completed': True, 'n': len(samples)})
    print(json.dumps(decision, indent=2))
if __name__ == '__main__':
    main()
