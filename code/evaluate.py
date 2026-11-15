"""Evaluate newly generated predictions or an existing score archive."""
import argparse
from pathlib import Path
import numpy as np
from sonar.data import DATASETS, load_manifest, read_json, write_json
from sonar.metrics import evaluate


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', choices=DATASETS, required=True)
    p.add_argument('--predictions', type=Path, required=True)
    p.add_argument('--manifest', type=Path)
    p.add_argument('--classes', type=Path)
    p.add_argument('--reference-samples', type=Path, help='Optional original per-clip ranks for exact score ties in historical archives.')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    root = Path(__file__).resolve().parents[1]
    classes = read_json(args.classes or root / 'manifests' / args.dataset / 'classes.json')
    rows = load_manifest(args.manifest or root / 'manifests' / args.dataset / 'samples.json', classes)
    reference = read_json(args.reference_samples) if args.reference_samples else None
    if reference is not None:
        if len(reference) != len(rows) or any(a['sample_index'] != b['sample_index'] or a['true_indices'] != b['true_indices']
                                               for a, b in zip(reference, rows)):
            raise ValueError('Original prediction records do not align with the evaluation manifest.')
    with np.load(args.predictions, allow_pickle=False) as a:
        scores = a['scores']
        if scores.shape[0] != len(rows) or scores.shape[-1] != len(classes): raise ValueError('Score shape differs from the manifest.')
        if 'classes' in a and a['classes'].tolist() != classes: raise ValueError('Candidate class order differs.')
        if 'sample_indices' in a and a['sample_indices'].tolist() != [r['sample_index'] for r in rows]: raise ValueError('Sample order differs.')
        names = a['methods'].tolist() if scores.ndim == 3 else [str(a['method'].item()) if 'method' in a else 'method']
        cube = scores if scores.ndim == 3 else scores[:, None, :]
        results = []
        for mi, name in enumerate(names):
            order = (a['orders'][:, mi, :] if a['orders'].ndim == 3 else a['orders']) if 'orders' in a else None
            ranks = [r['ranks'][name] for r in reference] if reference is not None else None
            metrics, per_class, _ = evaluate(cube[:, mi], [r['true_indices'] for r in rows], DATASETS[args.dataset][1],
                                              orders=order, reference_ranks=ranks)
            results.append(dict(method=name, metrics=metrics, per_class=per_class))
    write_json(args.output, dict(dataset=args.dataset, n=len(rows), classes=classes, results=results))
    print(f'Evaluated {len(rows)} clips; saved {args.output}')


if __name__ == '__main__':
    main()
