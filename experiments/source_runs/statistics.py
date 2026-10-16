from saki.paths import release_path
"""Recompute paper statistics using corrected FSD50K/AudioSet and unchanged peers."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.stats import binomtest
ROOT = Path(release_path('${RUN_ROOT}'))
OUT = ROOT / '31_corrected_aakv_cache/statistics'
DATASETS = [('01_ESC50', 'ESC-50', 2000), ('02_UrbanSound8K', 'UrbanSound8K', 8732), ('03_FSD50K', 'FSD50K', 10231), ('05_AudioSet', 'AudioSet', 17233), ('06_TUT2017', 'TUT2017', 6300)]
PRIMARY = ('I_iKnow', 'TFS_Final', 'Final-TFS - Frozen-iKnow')
CONTRASTS = [('I_iKnow', 'T_AAKV', 'T only - iKnow'), ('I_iKnow', 'F_Fusion', 'F only - iKnow'), ('I_iKnow', 'S_Selector', 'S only - iKnow'), ('TF', 'TFS_Final', 'add S to TF'), ('TS', 'TFS_Final', 'add F to TS'), ('FS', 'TFS_Final', 'add T to FS')]

def read_rows(code):
    if code in ('03_FSD50K', '05_AudioSet'):
        source = ROOT / '31_corrected_aakv_cache/factorial/test' / code
    elif code == '06_TUT2017':
        source = ROOT / '28_TUT2017_full6300_rerun/merged_test' / code
    else:
        source = ROOT / '25_final_experiment_alpha03_Nr5/test' / code
    rows = json.loads((source / 'samples.json').read_text(encoding='utf-8'))
    return (source, rows)

def bootstrap(dh, dr, seed, reps=10000, chunk=250):
    rng = np.random.default_rng(seed)
    hs, rs = ([], [])
    for start in range(0, reps, chunk):
        idx = rng.integers(0, len(dh), size=(min(chunk, reps - start), len(dh)))
        hs.append(dh[idx].mean(axis=1) * 100)
        rs.append(dr[idx].mean(axis=1) * 100)
    return (np.percentile(np.concatenate(hs), [2.5, 97.5]), np.percentile(np.concatenate(rs), [2.5, 97.5]))

def holm(values):
    p = np.asarray(values, float)
    order = np.argsort(p)
    q = np.maximum.accumulate((len(p) - np.arange(len(p))) * p[order])
    out = np.empty_like(q)
    out[order] = np.minimum(q, 1)
    return out

def compare(name, rows, baseline, method, label, with_transitions=False):
    a = np.asarray([x['ranks'][baseline] for x in rows], float)
    b = np.asarray([x['ranks'][method] for x in rows], float)
    ah, bh = (a == 1, b == 1)
    rescued = int(np.sum(~ah & bh))
    harmed = int(np.sum(ah & ~bh))
    discordant = rescued + harmed
    p = float(binomtest(min(rescued, harmed), discordant, 0.5).pvalue) if discordant else 1.0
    dh = bh.astype(float) - ah.astype(float)
    dr = 1 / b - 1 / a
    seed = 20260911 + int(hashlib.sha256((name + '|' + label).encode()).hexdigest()[:8], 16)
    hci, rci = bootstrap(dh, dr, seed)
    result = {'dataset': name, 'n': len(rows), 'comparison': label, 'baseline': baseline, 'method': method, 'delta_Hit@1_pp': float(dh.mean() * 100), 'Hit@1_CI95_low': float(hci[0]), 'Hit@1_CI95_high': float(hci[1]), 'wrong_to_correct': rescued, 'correct_to_wrong': harmed, 'net_rescued': rescued - harmed, 'McNemar_exact_p': p, 'delta_MRR_pp': float(dr.mean() * 100), 'MRR_CI95_low': float(rci[0]), 'MRR_CI95_high': float(rci[1])}
    transitions = []
    if with_transitions:
        for i, row in enumerate(rows):
            state = 'wrong_to_correct' if not ah[i] and bh[i] else 'correct_to_wrong' if ah[i] and (not bh[i]) else 'both_correct' if ah[i] else 'both_wrong'
            transitions.append({'dataset': name, 'sample_index': row['sample_index'], 'audio_path': row['audio_path'], 'baseline_rank': int(a[i]), 'final_rank': int(b[i]), 'Hit@1_transition': state, 'delta_reciprocal_rank': float(dr[i])})
    return (result, transitions)

def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

def main():
    OUT.mkdir(exist_ok=True)
    primary, contrasts, transitions, sources = ([], [], [], {})
    for code, name, n in DATASETS:
        source, rows = read_rows(code)
        assert len(rows) == n, (name, len(rows), n)
        sources[name] = str(source)
        p, t = compare(name, rows, *PRIMARY, with_transitions=True)
        primary.append(p)
        transitions.extend(t)
        for a, b, label in CONTRASTS:
            x, _ = compare(name, rows, a, b, label)
            contrasts.append(x)
    for family in (primary, contrasts):
        for row, q in zip(family, holm([x['McNemar_exact_p'] for x in family])):
            row['McNemar_Holm_p_family'] = float(q)
            row['McNemar_significant_Holm_0.05'] = bool(q < 0.05)
            row['Hit@1_CI_excludes_zero'] = bool(row['Hit@1_CI95_low'] > 0 or row['Hit@1_CI95_high'] < 0)
            row['MRR_CI_excludes_zero'] = bool(row['MRR_CI95_low'] > 0 or row['MRR_CI95_high'] < 0)
    write_csv(OUT / '01_primary_final_vs_iknow.csv', primary)
    write_csv(OUT / '02_module_contrasts.csv', contrasts)
    write_csv(OUT / '03_primary_sample_transitions.csv', transitions)
    (OUT / 'protocol.json').write_text(json.dumps({'sources': sources, 'paired_bootstrap_repetitions': 10000, 'primary_Holm_family': 5, 'contrasts_Holm_family': 30}, indent=2), encoding='utf-8')
    print(json.dumps({'primary': primary, 'n_contrasts': len(contrasts)}, indent=2))
if __name__ == '__main__':
    main()
