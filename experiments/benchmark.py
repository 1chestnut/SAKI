"""Serial online latency: CLAP, iKnow dagger and SoNAR with 47/30/20 relations."""
import argparse
import gc
import importlib.metadata
import json
import platform
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from saki.data import DATASETS, configure, load_manifest, resolve_audio, seed_all, write_json
from saki.runtime import Branch, PreparedInputs, load_clap
from . import _timing_core as reference

CONFIGURATIONS = [Branch('CLAP', selection='base'),
                  Branch('iKnow_dagger', text='direct', selection='fixed_texts', fusion='joint'),
                  *[Branch(f'SoNAR_R{n}', pool_size=n) for n in (47, 30, 20)]]


def timed(fn, device):
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    result = fn()
    torch.cuda.synchronize(device)
    return result, time.perf_counter() - start, torch.cuda.max_memory_allocated(device)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', choices=DATASETS, required=True)
    p.add_argument('--prepared', type=Path, required=True)
    p.add_argument('--audio-root', type=Path, required=True)
    p.add_argument('--weights', type=Path, required=True)
    p.add_argument('--manifest', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--cpus', type=int, nargs=2, required=True, help='Use the same two available CPU cores for every dataset and configuration.')
    args = p.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    device = torch.device(args.device)
    if device.type != 'cuda': raise ValueError('The reported protocol uses CUDA; CPU timing is a different experiment.')
    torch.cuda.set_device(device)
    configure(threads=2, cpus=args.cpus)
    data = PreparedInputs(args.prepared)
    manifest = load_manifest(args.manifest or Path(__file__).resolve().parents[1] / 'manifests' / args.dataset / 'samples.json', data.classes)
    if len(manifest) < 200: raise ValueError('The timing protocol requires 200 unique audio clips.')
    positions = np.linspace(0, len(manifest) - 1, 200, dtype=int).tolist()
    chosen = [manifest[i] for i in positions]
    paths = [resolve_audio(r, args.audio_root) for r in chosen]
    for path in paths:
        # Populate the file cache equally before any timed configuration.
        with path.open('rb') as stream:
            while stream.read(1024 * 1024): pass
    model = load_clap(args.weights, device)
    args.output.mkdir(parents=True)
    repetitions, sample_records, first_predictions = [], [], {}
    for repeat in range(5):
        rotated = CONFIGURATIONS[repeat:] + CONFIGURATIONS[:repeat]
        for branch in rotated:
            active = data.activate(branch, device)
            def scoring(audio):
                method = branch.name if branch.name in ('CLAP', 'iKnow_dagger') else 'SoNAR'
                return reference.score_and_rank(method, audio, active.labels, active.fixed_embeddings,
                                                active.fixed_indices, active.relations, active.records, active.embeddings)
            def encoding(path):
                return F.normalize(torch.as_tensor(model.get_audio_embeddings([str(path)]),
                                                   device=device, dtype=torch.float32), dim=-1)
            for row, path in zip(chosen[:20], paths[:20]):
                seed_all(row['seed'])
                warm_audio = encoding(path)
                scoring(warm_audio)
            del warm_audio
            torch.cuda.synchronize(device)
            total_audio = total_score = 0.0
            peak = 0
            predicted = []
            for row, path in zip(chosen, paths):
                seed_all(row['seed'])
                audio, t1, m1 = timed(lambda: encoding(path), device)
                scored, t2, m2 = timed(lambda: scoring(audio), device)
                prediction = int(scored[1][0, 0].cpu())
                predicted.append(prediction)
                total_audio += t1; total_score += t2; peak = max(peak, m1, m2)
                sample_records.append(dict(configuration=branch.name, repeat=repeat + 1,
                    sample_index=row['sample_index'], audio_processing_encoding_ms=t1 * 1000,
                    scoring_ranking_ms=t2 * 1000, online_latency_ms=(t1 + t2) * 1000))
                del audio, scored
            if branch.name not in first_predictions: first_predictions[branch.name] = predicted
            agreement = float(np.mean(np.asarray(predicted) == np.asarray(first_predictions[branch.name])))
            if agreement < .995: raise RuntimeError('Top-1 agreement across repeats is below the accepted 99.5% threshold.')
            record = dict(configuration=branch.name, repeat=repeat + 1, n_samples=200,
                online_latency_ms=1000 * (total_audio + total_score) / 200,
                audio_processing_encoding_ms=1000 * total_audio / 200, scoring_ranking_ms=1000 * total_score / 200,
                peak_gpu_mib=peak / 2**20, offline_cache_mib=active.cache_mib,
                top1_agreement_vs_first_repeat=agreement)
            repetitions.append(record)
            write_json(args.output / 'timing_repeats.json', repetitions)
            print(f"{args.dataset}: {branch.name}, repeat {repeat + 1}, {record['online_latency_ms']:.3f} ms/clip", flush=True)
            del active
            gc.collect(); torch.cuda.empty_cache()
    fields = ('online_latency_ms', 'audio_processing_encoding_ms', 'scoring_ranking_ms', 'peak_gpu_mib', 'offline_cache_mib')
    summaries = []
    for branch in CONFIGURATIONS:
        records = [r for r in repetitions if r['configuration'] == branch.name]
        summary = dict(configuration=branch.name, **{k: float(np.mean([r[k] for r in records])) for k in fields})
        summary['throughput_samples_per_s'] = 1000 / summary['online_latency_ms']
        summaries.append(summary)
    versions = {}
    for package in ('torch', 'numpy', 'msclap', 'transformers', 'librosa', 'torchaudio'):
        try: versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError: pass
    write_json(args.output / 'timing_samples.json', sample_records)
    write_json(args.output / 'summary.json', dict(dataset=args.dataset, results=summaries))
    write_json(args.output / 'protocol.json', dict(dataset=args.dataset, batch_size=1, cpu_threads=2,
        cpu_affinity=args.cpus, n_samples=200, repetitions=5, warmup_clips=20,
        manifest_positions=positions, sample_indices=[r['sample_index'] for r in chosen],
        configuration_order='five cyclic rotations', cuda_synchronization='before and after both timed segments',
        online_includes=['audio decoding and preprocessing', 'CLAP audio encoding', 'matching and relation scoring', 'selection and fusion', 'ranking'],
        offline_excludes=['graph retrieval', 'AAKV generation', 'text encoding'],
        full_test_classification='Run experiments.run --suite pools separately; the timing subset is not an accuracy evaluation.',
        gpu_name=torch.cuda.get_device_name(device), cuda=torch.version.cuda, python=platform.python_version(), versions=versions))


if __name__ == '__main__':
    main()
