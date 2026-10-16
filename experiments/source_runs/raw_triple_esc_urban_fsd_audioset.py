from saki.paths import release_path
"""Task25(g): RawTriple + frozen Fusion + frozen Selector control."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
os.environ['PYTHONHASHSEED'] = '42'
import argparse
import hashlib
import importlib.util
import json
import math
import random
from collections import Counter, OrderedDict
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
ROOT = Path(release_path('${RUN_ROOT}'))
TASK = ROOT / '25g_RawTriple_FS_control'
SOURCE25 = ROOT / '25_final_experiment_alpha03_Nr5' / 'test'

def seed_all(value):
    random.seed(value)
    np.random.seed(value % (2 ** 32 - 1))
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)

def load_module(path):
    spec = importlib.util.spec_from_file_location('task25g_runtime', str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def nlse(values):
    return (torch.logsumexp(100.0 * values, dim=0) - math.log(values.numel())) / 100.0

def hier(base_score, evidence_scores):
    return base_score if not evidence_scores.numel() else 0.3 * base_score + 0.7 * nlse(evidence_scores.reshape(-1))

def rank(scores, true_indices):
    order = torch.argsort(scores, descending=True).cpu().numpy()
    return min((int(np.where(order == int(y))[0][0]) + 1 for y in true_indices))

def metrics(ranks):
    values = np.asarray(ranks, dtype=float)
    return {'Hit@1': float(100 * np.mean(values <= 1)), 'Hit@3': float(100 * np.mean(values <= 3)), 'Hit@5': float(100 * np.mean(values <= 5)), 'MRR': float(100 * np.mean(1.0 / values))}

def relation_order(spaces, relations):
    prediction, margin = ({}, {})
    for relation in relations:
        top = torch.topk(spaces[relation], k=min(2, spaces[relation].numel())).values
        prediction[relation] = int(torch.argmax(spaces[relation]))
        margin[relation] = float(top[0] - top[1]) if top.numel() > 1 else 0.0
    votes = Counter(prediction.values())
    max_votes = max(votes.values())
    candidates = sorted((c for c, count in votes.items() if count == max_votes))
    consensus_class = max(candidates, key=lambda c: (sum((margin[r] for r in relations if prediction[r] == c)), -c))
    margin_order = sorted(relations, key=lambda r: (-margin[r], r))
    supporting = sorted((r for r in relations if prediction[r] == consensus_class), key=lambda r: (-margin[r], r))
    return (supporting + [r for r in margin_order if r not in supporting], consensus_class, max_votes)

def apply_selected(base, top_classes, evidence, chosen):
    output = base.clone()
    for class_id in top_classes:
        merged = OrderedDict()
        for relation in chosen:
            for tail, score in evidence[class_id][relation]:
                merged.setdefault(tail, score)
        values = torch.stack(list(merged.values())) if merged else torch.empty(0, device=base.device)
        output[class_id] = hier(base[class_id], values)
    return output

def main(dataset):
    output = TASK / dataset
    output.mkdir(parents=True, exist_ok=True)
    source = ROOT / '12_shared-abcd-statistics' / dataset
    runtime = load_module(Path(__file__).parent / 'runtime' / dataset / 'runtime.py')
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
    frozen = torch.load(source / 'text_embeddings.pt', map_location='cpu')
    label_embeddings = frozen['labels'].to(device)
    relation_cache = torch.load(ROOT / '18a_relation_oracle_audit' / dataset / 'results' / 'static_relation_cache.pt', map_location='cpu')
    relations = relation_cache['relations']
    triples = []
    triple_index = {}
    records = {relation: {} for relation in relations}
    for relation in relations:
        for class_id, items in relation_cache['tails'][relation].items():
            values = []
            for tail, direct_prompt in items:
                head = direct_prompt.split(', ', 1)[0]
                raw_text = f'{head} {relation} {tail}'
                key = (head, relation, tail)
                if key not in triple_index:
                    triple_index[key] = len(triples)
                    triples.append(raw_text)
                values.append((tail, triple_index[key]))
            records[relation][int(class_id)] = values
    chunks = []
    for start in range(0, len(triples), 128):
        chunks.append(F.normalize(runtime.get_safe_text_embeddings(clap, triples[start:start + 128], device), dim=-1))
    raw_embeddings = torch.cat(chunks, dim=0)
    raw_ranks, rows, score_rows = ([], [], [])
    selection_counts = Counter()
    for i, sample in enumerate(tqdm(samples, desc=dataset, mininterval=10)):
        sample_seed = int(hashlib.sha256(('42|' + sample['audio_path']).encode()).hexdigest()[:8], 16)
        seed_all(sample_seed)
        audio = torch.load(source / 'audio_cache' / f'{i:06d}.pt', map_location=device)
        base = (audio @ label_embeddings.T).squeeze(0)
        top_classes = torch.argsort(base, descending=True)[:5].tolist()
        similarities = (audio @ raw_embeddings.T).squeeze(0)
        evidence = {class_id: {} for class_id in top_classes}
        for class_id in top_classes:
            for relation in relations:
                evidence[class_id][relation] = [(tail, similarities[index]) for tail, index in records[relation].get(class_id, [])]
        single_relation_scores = {relation: apply_selected(base, top_classes, evidence, [relation]) for relation in relations}
        order, consensus_class, votes = relation_order(single_relation_scores, relations)
        chosen = order[:5]
        raw_fs = apply_selected(base, top_classes, evidence, chosen)
        current_rank = rank(raw_fs, sample['true_indices'])
        raw_ranks.append(current_rank)
        selection_counts.update(chosen)
        rows.append({'sample_index': i, 'audio_path': sample['audio_path'], 'true_indices': [int(x) for x in sample['true_indices']], 'seed': sample_seed, 'selected_relations': chosen, 'consensus_class': consensus_class, 'votes': votes, 'rank': current_rank})
        score_rows.append(raw_fs.cpu().numpy())
    result = metrics(raw_ranks)
    np.savez_compressed(output / 'predictions.npz', scores=np.asarray(score_rows), method='Raw-FS')
    (output / 'samples.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding='utf-8')
    (output / 'metrics.json').write_text(json.dumps({'Raw-FS': result}, indent=2), encoding='utf-8')
    (output / 'metrics.csv').write_text(f"method,Hit@1,Hit@3,Hit@5,MRR\nRaw-FS,{result['Hit@1']:.8f},{result['Hit@3']:.8f},{result['Hit@5']:.8f},{result['MRR']:.8f}\n", encoding='utf-8')
    protocol = {'source': 'Task25 frozen configuration', 'only_changed_variable': "knowledge text: Direct/AAKV -> raw 'head relation tail'", 'formula': 'hierarchical fusion with normalized LSE', 'alpha': 0.3, 'Nr': 5, 'K': 5, 'M': 3, 'TopP': 'All/disabled', 'hop': 1, 'seed': 42, 'labels_used_by_selector': False, 'selection_counts': dict(selection_counts)}
    (output / 'protocol.json').write_text(json.dumps(protocol, ensure_ascii=False, indent=2), encoding='utf-8')
    (output / 'progress.json').write_text(json.dumps({'completed': True, 'n': len(samples)}, indent=2), encoding='utf-8')
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset')
    main(parser.parse_args().dataset)
