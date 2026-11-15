"""Paired bootstrap intervals, exact McNemar tests and Holm adjustment."""
import hashlib
import numpy as np
from scipy.stats import binomtest


def holm(values):
    p = np.asarray(values, dtype=float)
    order = np.argsort(p)
    adjusted = np.maximum.accumulate((len(p) - np.arange(len(p))) * p[order])
    out = np.empty_like(adjusted)
    out[order] = np.minimum(adjusted, 1)
    return out


def compare(baseline_ranks, method_ranks, dataset_name, repetitions=10000):
    a, b = np.asarray(baseline_ranks, float), np.asarray(method_ranks, float)
    if a.shape != b.shape or a.ndim != 1 or not a.size or np.any(a < 1) or np.any(b < 1):
        raise ValueError("Paired ranks must be positive and have the same sample order.")
    ah, bh = a == 1, b == 1
    corrected, degraded = int(np.sum(~ah & bh)), int(np.sum(ah & ~bh))
    discordant = corrected + degraded
    p = float(binomtest(min(corrected, degraded), discordant, 0.5).pvalue) if discordant else 1.0
    dh, dr = bh.astype(float) - ah.astype(float), 1 / b - 1 / a
    label = 'Final-TFS - Frozen-iKnow'  # Preserve the accepted bootstrap RNG identity.
    seed = 20260911 + int(hashlib.sha256((dataset_name + '|' + label).encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    hs, rs = [], []
    for start in range(0, repetitions, 250):
        indices = rng.integers(0, len(a), size=(min(250, repetitions - start), len(a)))
        hs.append(dh[indices].mean(axis=1) * 100)
        rs.append(dr[indices].mean(axis=1) * 100)
    return dict(n=len(a), delta_Hit1_pp=float(dh.mean() * 100), delta_MRR_pp=float(dr.mean() * 100),
                Hit1_CI95=np.percentile(np.concatenate(hs), [2.5, 97.5]).tolist(),
                MRR_CI95=np.percentile(np.concatenate(rs), [2.5, 97.5]).tolist(),
                wrong_to_correct=corrected, correct_to_wrong=degraded, McNemar_exact_p=p,
                correction_rate_percent=100 * corrected / int((~ah).sum()) if (~ah).any() else None,
                degradation_rate_percent=100 * degraded / int(ah.sum()) if ah.any() else None)
