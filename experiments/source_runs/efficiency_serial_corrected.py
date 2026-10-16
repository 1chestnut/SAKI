from saki.paths import release_path
"""Serial version-31 profile: fixed GPU/CPU affinity, batch size 1, all repeats retained."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
os.environ['PYTHONHASHSEED'] = '42'
import argparse, csv, gc, hashlib, importlib.util, json, math, random, time
from collections import Counter, OrderedDict
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
ROOT = Path(release_path('${RUN_ROOT}'))
TASK = Path(os.environ.get('P1_2_TASK_ROOT', str(ROOT / '31_corrected_aakv_cache/online_efficiency_serial_20261008/runs/attempt_1')))
AAKV = ROOT / '31_corrected_aakv_cache/aakv_all.json'
METHODS = ('CLAP', 'iKnow_dagger', 'SAKI')
K, M, KAPPA, ALPHA, NR = (5, 3, 100.0, 0.3, 5)
DS_DIR = {'ESC-50': '01_ESC50', 'UrbanSound8K': '02_UrbanSound8K', 'FSD50K': '03_FSD50K', 'AudioSet': '05_AudioSet', 'TUT2017': '06_TUT2017'}

def seed_all(v=42):
    random.seed(v)
    np.random.seed(v)
    torch.manual_seed(v)
    torch.cuda.manual_seed_all(v)

def load_module(path):
    spec = importlib.util.spec_from_file_location('rt25a', str(path))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m

def nlse(v):
    return (torch.logsumexp(KAPPA * v, 0) - math.log(v.numel())) / KAPPA

def joint(b, e):
    return b if not e.numel() else nlse(torch.cat((b.reshape(1), e.reshape(-1))))

def hier(b, e):
    return b if not e.numel() else ALPHA * b + (1 - ALPHA) * nlse(e.reshape(-1))

def relation_order(spaces, relations):
    pred = {}
    margin = {}
    for r in relations:
        vals = torch.topk(spaces[r], k=min(2, spaces[r].numel())).values
        pred[r] = int(torch.argmax(spaces[r]))
        margin[r] = float(vals[0] - vals[1]) if vals.numel() > 1 else 0.0
    votes = Counter(pred.values())
    mv = max(votes.values())
    cand = sorted((c for c, n in votes.items() if n == mv))
    cc = max(cand, key=lambda c: (sum((margin[r] for r in relations if pred[r] == c)), -c))
    mo = sorted(relations, key=lambda r: (-margin[r], r))
    support = sorted((r for r in relations if pred[r] == cc), key=lambda r: (-margin[r], r))
    return support + [r for r in mo if r not in support]

def final_one(base, audio, top, relations, per, ae):
    similarity = (audio @ ae.T).reshape(-1)
    spaces = {}
    for r in relations:
        z = base.clone()
        for ci in top:
            ids = [idx for tail, idx in per[ci][r]]
            ev = audio.new_empty(0) if not ids else torch.stack([similarity[idx] for idx in ids])
            z[ci] = hier(base[ci], ev)
        spaces[r] = z
    chosen = relation_order(spaces, relations)[:NR]
    z = base.clone()
    for ci in top:
        merged = OrderedDict()
        for r in chosen:
            for tail, idx in per[ci][r]:
                merged.setdefault(tail, idx)
        ids = list(merged.values())
        ev = audio.new_empty(0) if not ids else torch.stack([similarity[idx] for idx in ids])
        z[ci] = hier(base[ci], ev)
    return z

def run_method(method, audios, label_emb, frozen_emb, frozen_idx, relations, per_class, aakv_emb):
    outputs = []
    for audio in audios:
        base = (audio.reshape(1, -1) @ label_emb.T).squeeze(0)
        top = torch.argsort(base, descending=True)[:K].tolist()
        if method == 'CLAP':
            z = base
        elif method == 'iKnow_dagger':
            z = base.clone()
            for ci in top:
                ev = (audio.reshape(1, -1) @ frozen_emb[frozen_idx[ci]['direct']].T).reshape(-1)
                z[ci] = joint(base[ci], ev)
        else:
            z = final_one(base, audio.reshape(1, -1), top, relations, per_class, aakv_emb)
        outputs.append(z)
    return torch.stack(outputs)

def sync():
    torch.cuda.synchronize()

def score_and_rank(*args):
    scores = run_method(*args)
    return (scores, torch.argsort(scores, dim=1, descending=True))

def write_rows(path, rows):
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    tmp.replace(path)

def progress(out, **kwargs):
    tmp = out / 'progress.json.tmp'
    tmp.write_text(json.dumps(kwargs, ensure_ascii=False, indent=2), encoding='utf-8')
    tmp.replace(out / 'progress.json')

def guard_gpu():
    import subprocess
    uuid = os.environ.get('PROFILE_GPU_UUID')
    if not uuid:
        return
    data = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader,nounits'], text=True)
    other = []
    for line in data.splitlines():
        cols = [x.strip() for x in line.split(',')]
        if len(cols) == 2 and cols[0] == uuid and (int(cols[1]) != os.getpid()):
            other.append(int(cols[1]))
    if other:
        raise RuntimeError('Foreign GPU process detected; measurement rejected: ' + str(other))

def timed(fn):
    sync()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    value = fn()
    sync()
    elapsed = time.perf_counter() - t0
    return (value, elapsed, torch.cuda.max_memory_allocated())

def batches(items, n):
    for i in range(0, len(items), n):
        yield items[i:i + n]

def final_samples(ds):
    if ds == 'TUT2017':
        path = ROOT / '28_TUT2017_full6300_rerun/main/test/06_TUT2017/samples.json'
        expected = 6300
    elif ds in ('FSD50K', 'AudioSet'):
        path = ROOT / '31_corrected_aakv_cache/factorial/test' / DS_DIR[ds] / 'samples.json'
        expected = {'FSD50K': 10231, 'AudioSet': 17233}[ds]
    else:
        path = ROOT / '25_final_experiment_alpha03_Nr5/test' / DS_DIR[ds] / 'samples.json'
        expected = {'ESC-50': 2000, 'UrbanSound8K': 8732, 'FSD50K': 10231, 'AudioSet': 17233}[ds]
    rows = json.loads(path.read_text(encoding='utf-8'))
    if isinstance(rows, dict):
        rows = rows.get('samples', rows.get('records', rows))
    if len(rows) != expected:
        raise RuntimeError(f'{ds}: final manifest count {len(rows)} != {expected}')
    ids = np.linspace(0, len(rows) - 1, 200, dtype=int).tolist()
    if len(set(ids)) != 200:
        raise RuntimeError('duplicate profile sample indices')
    return (path, [rows[i] for i in ids], ids, hashlib.sha256(path.read_bytes()).hexdigest())

@torch.inference_mode()
def main(ds, smoke=False):
    seed_all()
    out = TASK / ds
    out.mkdir(parents=True, exist_ok=True)
    src = ROOT / '12_shared-abcd-statistics' / DS_DIR[ds]
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    progress(out, completed=False, phase='offline_preparation', pid=os.getpid())
    rt = load_module(src / 'frozen_inputs/runtime.py')
    dev = rt.DEVICE
    clap = rt.CLAP(version='2023', use_cuda=True)
    clap.clap.eval()
    manifest, chosen, ids, manifest_sha = final_samples(ds)
    if smoke:
        chosen = chosen[:2]
        ids = ids[:2]
    count = len(chosen)
    reps = 1 if smoke else 5
    batch_sizes = (1,)
    for sample in chosen:
        with open(sample['audio_path'], 'rb') as audio_file:
            while audio_file.read(1024 * 1024):
                pass
    frozen = torch.load(src / 'text_embeddings.pt', map_location='cpu')
    le = frozen['labels'].to(dev)
    fe_cpu = frozen['evidence']
    fi = frozen['indices']
    cache = torch.load(ROOT / f'18a_relation_oracle_audit/{DS_DIR[ds]}/results/static_relation_cache.pt', map_location='cpu')
    relations = cache['relations']
    texts = json.loads(AAKV.read_text(encoding='utf-8'))['texts']
    keys = []
    seen = set()
    per = {}
    for r in relations:
        for ci, items in cache['tails'][r].items():
            per.setdefault(int(ci), {}).setdefault(r, [])
            for tail, prompt in items:
                head = cache['labels'][int(ci)]
                assert prompt.startswith(head + ', '), (ds, ci, head, prompt)
                key = '\t'.join((head, r, tail))
                if key not in seen:
                    seen.add(key)
                    keys.append(key)
                per[int(ci)][r].append((tail, key))
    for ci in range(le.shape[0]):
        per.setdefault(ci, {})
        for r in relations:
            per[ci].setdefault(r, [])
    missing = [k for k in keys if k not in texts]
    if missing:
        raise RuntimeError(f'Missing {len(missing)} corrected AAKV keys')
    key_id = {k: i for i, k in enumerate(keys)}
    per = {ci: {r: [(tail, key_id[key]) for tail, key in vals] for r, vals in rv.items()} for ci, rv in per.items()}
    chunks = []
    for i in range(0, len(keys), 128):
        chunks.append(F.normalize(rt.get_safe_text_embeddings(clap, [texts[k] for k in keys[i:i + 128]], dev), dim=-1))
    ae_cpu = torch.cat(chunks).cpu()
    del chunks
    cache_out = out / 'offline_cache'
    cache_out.mkdir(exist_ok=True)
    torch.save({'labels': le.cpu(), 'aakv': ae_cpu, 'relation_indices': per}, cache_out / 'saki_evidence.pt')
    gc.collect()
    torch.cuda.empty_cache()
    print(f'{ds}: offline evidence cache ready; {len(keys)} AAKV texts; {count} fixed samples', flush=True)
    label_bytes = le.numel() * le.element_size()
    index_bytes = len(json.dumps(fi, ensure_ascii=False, default=str).encode('utf-8'))
    relation_bytes = len(json.dumps(per, ensure_ascii=False, default=str).encode('utf-8'))
    cache_bytes = {'CLAP': label_bytes, 'iKnow_dagger': label_bytes + fe_cpu.numel() * fe_cpu.element_size() + index_bytes, 'SAKI': label_bytes + ae_cpu.numel() * ae_cpu.element_size() + relation_bytes}
    visible_gpu = os.environ.get('CUDA_VISIBLE_DEVICES', 'unspecified')
    gpu_name = torch.cuda.get_device_name(0)
    rows = []
    reference_top1 = {}
    for bs in batch_sizes:
        for rep in range(reps):
            order = METHODS[rep % 3:] + METHODS[:rep % 3]
            for method in order:
                guard_gpu()
                progress(out, completed=False, phase='timing', pid=os.getpid(), batch_size=bs, repeat=rep + 1, method=method, finished_rows=len(rows))
                fe = fe_cpu.to(dev) if method == 'iKnow_dagger' else None
                ae = ae_cpu.to(dev) if method == 'SAKI' else None
                warm = chosen[:2 if smoke else 20]
                for group in batches(warm, bs):
                    a = F.normalize(rt.to_tensor(clap.get_audio_embeddings([x['audio_path'] for x in group])).to(dev).float(), dim=-1)
                    score_and_rank(method, a, le, fe, fi, relations, per, ae)
                sync()
                total_e2e = 0.0
                total_stage = 0.0
                peak = 0
                n = 0
                full_hash = hashlib.sha256()
                top1 = []
                reference_matches = []
                for group in batches(chosen, bs):
                    paths = [x['audio_path'] for x in group]
                    batch_seed = int(hashlib.sha256(('42|' + '|'.join(paths)).encode()).hexdigest()[:8], 16)
                    seed_all(batch_seed)
                    audio, e2e_encode, mem1 = timed(lambda: F.normalize(rt.to_tensor(clap.get_audio_embeddings(paths)).to(dev).float(), dim=-1))
                    scored, stage, mem2 = timed(lambda: score_and_rank(method, audio, le, fe, fi, relations, per, ae))
                    scores, rank_order = scored
                    total_e2e += e2e_encode + stage
                    total_stage += stage
                    peak = max(peak, mem1, mem2)
                    n += len(group)
                    order = rank_order.cpu().numpy()
                    full_hash.update(order.tobytes())
                    top1.extend(order[:, 0].tolist())
                    if bs == 1:
                        refkey = {'CLAP': 'CLAP', 'iKnow_dagger': 'I_iKnow', 'SAKI': 'TFS_Final'}[method]
                        for sample, ordered in zip(group, order):
                            actual = min((int(np.where(ordered == int(y))[0][0]) + 1 for y in sample['true_indices']))
                            expected_rank = sample.get('ranks', {}).get(refkey)
                            if expected_rank is not None:
                                reference_matches.append(actual == expected_rank)
                    del scored, scores, rank_order, audio
                key = (bs, method)
                digest = full_hash.hexdigest()
                top1_arr = np.asarray(top1, dtype=np.int64)
                top1_digest = hashlib.sha256(top1_arr.tobytes()).hexdigest()
                if key in reference_top1:
                    agreement = float(np.mean(reference_top1[key] == top1_arr))
                    if agreement < 0.995:
                        raise RuntimeError(f'Top-1 repeat agreement below 99.5%: {key}={agreement:.6f}')
                else:
                    reference_top1[key] = top1_arr.copy()
                    agreement = 1.0
                guard_gpu()
                refagreement = float(np.mean(reference_matches)) if reference_matches else None
                if bs == 1 and method == 'SAKI' and (refagreement is not None) and (refagreement < 0.99):
                    raise RuntimeError(f'Corrected SAKI rank agreement below 99%: {refagreement}; verify live encoder/model inputs')
                rows.append({'dataset': ds, 'physical_gpu_id': visible_gpu, 'gpu_name': gpu_name, 'method': method, 'batch_size': bs, 'repeat': rep + 1, 'n_samples': n, 'end_to_end_ms_per_sample': 1000 * total_e2e / n, 'knowledge_stage_ms_per_sample': 1000 * total_stage / n, 'throughput_samples_per_s': n / total_e2e, 'peak_gpu_mib': peak / 2 ** 20, 'offline_cache_mib': cache_bytes[method] / 2 ** 20, 'top1_agreement_vs_first_repeat': agreement, 'top1_sha256': top1_digest, 'full_ranking_sha256_audit': digest, 'ground_truth_rank_agreement_vs_final_records': refagreement})
                write_rows(out / 'timing_repeats.csv', rows)
                print(f'{ds}: batch={bs} repeat={rep + 1} method={method} latency={1000 * total_e2e / n:.3f} ms/sample', flush=True)
                del fe, ae
                gc.collect()
                torch.cuda.empty_cache()
    with (out / 'timing_repeats.csv').open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    summary = []
    for bs in batch_sizes:
        for m in METHODS:
            rr = [x for x in rows if x['batch_size'] == bs and x['method'] == m]
            rec = {'dataset': ds, 'method': m, 'batch_size': bs, 'n_profiled': count, 'repeats': reps}
            for k in ('end_to_end_ms_per_sample', 'knowledge_stage_ms_per_sample', 'throughput_samples_per_s', 'peak_gpu_mib', 'offline_cache_mib'):
                a = np.asarray([x[k] for x in rr])
                rec[k + '_mean'] = float(a.mean())
                rec[k + '_std'] = float(a.std(ddof=1)) if len(a) > 1 else None
            rec['minimum_top1_agreement_vs_first_repeat'] = float(min((x['top1_agreement_vs_first_repeat'] for x in rr)))
            summary.append(rec)
    with (out / 'timing_summary.csv').open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)
    protocol = {'task': 'P1-2', 'dataset': ds, 'physical_gpu_id': visible_gpu, 'gpu_name': gpu_name, 'comparison': list(METHODS), 'sample_selection': '200 evenly spaced samples from final per-sample manifest; identical across methods', 'final_manifest': str(manifest), 'final_manifest_sha256': manifest_sha, 'sample_manifest_positions': ids, 'online_timing': {'end_to_end': 'live audio decode/preprocess/CLAP audio encoding + method-specific scoring and ranking', 'knowledge_stage': 'starts from current audio embedding; includes all method-specific matching, selection, fusion and ranking'}, 'offline_excluded': ['KG entity mapping', 'RotatE Top-M tail retrieval', 'AAKV text generation', 'all text embedding computation'], 'input_io_control': 'All 200 selected audio files are read once before model timing to reduce method-order-dependent cold-file-cache effects; model warm-up remains 20 samples per method, batch size and repeat.', 'offline_cache_size_definition': 'float32 candidate label embeddings plus method-specific precomputed evidence embeddings and serialized evidence index mapping; excludes model weights, source KG, raw audio and generator', 'fixed': {'model': 'same CLAP checkpoint', 'device': 'same physical GPU', 'K': K, 'M': M, 'kappa': KAPPA, 'alpha': ALPHA, 'Nr': NR, 'hop': 1, 'TopP': 'disabled', 'seed': 42}, 'fairness': ['all methods for a dataset use the same isolated physical GPU', 'different datasets may run concurrently on separate GPUs', 'GPU identity is recorded', 'same sample order', 'same immutable text caches', 'deterministic seed reset from each immutable audio-batch identity before live CLAP encoding', '20-sample warm-up', 'CUDA synchronization around every timed segment', 'rotated method order', 'five repeats', 'Top-1 agreement across live re-encoding repeats must be at least 99.5%; complete ranking hashes are retained as an audit because near-tied low-ranked classes can exchange order'], 'accuracy_source': 'Version31 for corrected FSD50K/AudioSet, final unchanged ESC50/US8K and TUT6300 manifests; subset accuracy is only an integrity audit', 'corrected_aakv_cache': str(AAKV), 'corrected_aakv_sha256': hashlib.sha256(AAKV.read_bytes()).hexdigest(), 'runtime_source': str(src / 'frozen_inputs/runtime.py'), 'runtime_sha256': hashlib.sha256((src / 'frozen_inputs/runtime.py').read_bytes()).hexdigest(), 'measurement_code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'versions': {'torch': torch.__version__, 'numpy': np.__version__, 'cuda': torch.version.cuda}, 'implementation_changes_from_legacy': ['full class-name/relation/tail AAKV keys', 'version31 full-evidence matching before relation scoring', 'final ranking included in scoring-stage timer'], 'smoke_only': smoke, 'repeats_actual': reps, 'n_actual': count, 'cpu_threads': 2, 'torch_interop_threads': 1, 'cpu_affinity': sorted(os.sched_getaffinity(0)), 'serial_protocol': 'one fixed GPU, datasets and methods sequential; primary batch size 1 only; all five repeats retained', 'cache_policy': 'same preliminary file-cache population as previous profile; no cached audio embeddings used for timing'}
    (out / 'protocol.json').write_text(json.dumps(protocol, ensure_ascii=False, indent=2), encoding='utf-8')
    progress(out, completed=True, phase='complete', n_profiled=count, repeats=reps, smoke_only=smoke, pid=os.getpid())
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('dataset')
    p.add_argument('--smoke', action='store_true')
    a = p.parse_args()
    try:
        main(a.dataset, a.smoke)
    except Exception as exc:
        failed = TASK / a.dataset
        failed.mkdir(parents=True, exist_ok=True)
        progress(failed, completed=False, phase='failed', error=str(exc), pid=os.getpid())
        raise
