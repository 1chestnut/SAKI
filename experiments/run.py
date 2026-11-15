"""Run main results, component/text/relation ablations or the alpha/N_r grid."""
import argparse
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
from sonar.data import DATASETS, configure, load_manifest, write_json
from sonar.metrics import evaluate
from sonar.runtime import PreparedInputs, encode_audio, load_clap
from .branches import branches, select_development_configuration


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', choices=DATASETS, required=True)
    p.add_argument('--prepared', type=Path, required=True)
    p.add_argument('--audio-root', type=Path, required=True)
    p.add_argument('--weights', type=Path, required=True)
    p.add_argument('--manifest', type=Path)
    p.add_argument('--suite', choices=('main', 'components', 'relations', 'text', 'pools', 'grid'), default='main')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--limit', type=int, help='Smoke test only; never a complete test-set evaluation.')
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError('Choose a new output directory.')
    configure()
    prepared = PreparedInputs(args.prepared)
    manifest_path = args.manifest or Path(__file__).resolve().parents[1] / 'manifests' / args.dataset / 'samples.json'
    rows = load_manifest(manifest_path, prepared.classes)
    if args.limit is not None:
        if args.limit < 1: raise ValueError('limit must be positive.')
        rows = rows[:args.limit]
    model = load_clap(args.weights, args.device)
    configs = branches(args.suite)
    # Encoding is shared across comparisons here. Use experiments.benchmark for online latency.
    score_rows = {b.name: [] for b in configs}
    order_rows = {b.name: [] for b in configs}
    meta_rows = {b.name: [] for b in configs}
    active = [prepared.activate(b, args.device) for b in configs]
    with torch.inference_mode():
        for pos, row in enumerate(rows):
            audio = encode_audio(model, row, args.audio_root, args.device)
            for scorer in active:
                z, meta = scorer.score(audio, row.get('relation_selection_seeds'))
                name = scorer.branch.name
                score_rows[name].append(z.cpu().numpy())
                order_rows[name].append(torch.argsort(z, descending=True).cpu().numpy())
                meta_rows[name].append(meta)
            if (pos + 1) % 100 == 0 or pos + 1 == len(rows):
                print(f'{args.dataset}: {pos + 1}/{len(rows)}', flush=True)
    args.output.mkdir(parents=True)
    summaries = []
    task = DATASETS[args.dataset][1]
    for branch in configs:
        scores, orders = np.asarray(score_rows[branch.name]), np.asarray(order_rows[branch.name])
        metrics, per_class, ranks = evaluate(scores, [r['true_indices'] for r in rows], task, orders=orders)
        np.savez_compressed(args.output / (branch.name + '.npz'), scores=scores, orders=orders, ranks=ranks,
                            sample_indices=np.asarray([r['sample_index'] for r in rows]),
                            classes=np.asarray(prepared.classes), method=branch.name)
        write_json(args.output / (branch.name + '_selection.json'), meta_rows[branch.name])
        write_json(args.output / (branch.name + '_per_class.json'), per_class)
        summaries.append(dict(method=branch.name, configuration=asdict(branch), metrics=metrics))
    result = dict(dataset=args.dataset, n=len(rows), suite=args.suite, smoke_only=args.limit is not None, results=summaries)
    if args.suite == 'relations':
        random_rows = [x['metrics'] for x in summaries if x['method'].startswith('Random_')]
        result['Random_Top5_three_seed_mean'] = {k: float(np.mean([x[k] for x in random_rows])) for k in random_rows[0]}
    if args.suite == 'grid':
        result['role'] = 'development selection' if args.dataset == '04_DCASE17_T4' else 'test-set sensitivity diagnostic only'
        if args.dataset == '04_DCASE17_T4' and args.limit is None:
            result['selected'] = select_development_configuration(summaries)['configuration']
    write_json(args.output / 'summary.json', result)


if __name__ == '__main__':
    main()
