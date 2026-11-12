"""Paired analysis across datasets, using newly computed per-clip ranks."""
import argparse
from pathlib import Path
import numpy as np
from saki.data import DATASETS, read_json, write_json
from saki.statistics import compare, holm


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--comparisons', type=Path, required=True,
                   help='JSON list: dataset, baseline_npz and method_npz, paths relative to this JSON file.')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    result = []
    for spec in read_json(args.comparisons):
        with np.load(args.comparisons.parent / spec['baseline_npz'], allow_pickle=False) as a, np.load(args.comparisons.parent / spec['method_npz'], allow_pickle=False) as b:
            if not np.array_equal(a['sample_indices'], b['sample_indices']): raise ValueError('The paired sample orders differ.')
            if not np.array_equal(a['classes'], b['classes']): raise ValueError('The candidate class orders differ.')
            name = DATASETS[spec['dataset']][0]
            result.append(dict(dataset=spec['dataset'], **compare(a['ranks'], b['ranks'], name)))
    for row, adjusted in zip(result, holm([r['McNemar_exact_p'] for r in result])):
        row['McNemar_Holm_p'] = float(adjusted)
    write_json(args.output, dict(bootstrap_repetitions=10000, Holm_family_size=len(result), results=result))


if __name__ == '__main__':
    main()
