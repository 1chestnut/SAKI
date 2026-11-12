"""Ranking metrics, Macro-F1 and non-interpolated per-class average precision."""
import warnings
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_fscore_support


def evaluate(scores, true_indices, task, orders=None, reference_ranks=None):
    scores = np.asarray(scores)
    if scores.ndim != 2 or not np.isfinite(scores).all() or len(scores) != len(true_indices):
        raise ValueError("Expected finite [clips, classes] scores aligned to the manifest.")
    n, c = scores.shape
    if n == 0:
        raise ValueError("Evaluation requires at least one audio clip.")
    y = np.zeros((n, c), dtype=np.uint8)
    for i, labels in enumerate(true_indices):
        if not labels or len(set(labels)) != len(labels) or any(not 0 <= int(k) < c for k in labels):
            raise ValueError("Invalid ground-truth indices.")
        y[i, labels] = 1
    order = np.argsort(-scores, axis=1, kind='stable') if orders is None else np.asarray(orders)
    if order.shape != scores.shape or not np.array_equal(np.sort(order, axis=1), np.tile(np.arange(c), (n, 1))):
        raise ValueError("Each ranking must be a permutation of candidate class indices.")
    ranked_scores = np.take_along_axis(scores, order, axis=1)
    if np.any(np.diff(ranked_scores, axis=1) > 0):
        raise ValueError("Provided ranking is inconsistent with the class scores.")
    ranks = np.argmax(np.take_along_axis(y, order, axis=1), axis=1) + 1
    if reference_ranks is not None:
        reference_ranks = np.asarray(reference_ranks, dtype=int)
        best = np.where(y != 0, scores, -np.inf).max(axis=1)
        low = 1 + np.sum(scores > best[:, None], axis=1)
        high = np.sum(scores >= best[:, None], axis=1)
        if reference_ranks.shape != (n,) or np.any((reference_ranks < low) | (reference_ranks > high)):
            raise ValueError("Stored ranks disagree with the score ordering beyond exact ties.")
        ranks = reference_ranks
    elif orders is None and any(np.any(np.diff(row) == 0) for row in ranked_scores):
        warnings.warn("No stored class order: exact score ties follow lower class index.", stacklevel=2)
    result = {f'Hit@{k}': float(100 * np.mean(ranks <= k)) for k in (1, 3, 5)}
    result['MRR'] = float(100 * np.mean(1 / ranks))
    per_class = {'positive_samples': y.sum(axis=0).astype(int).tolist()}
    if task == 'single-label':
        if not np.all(y.sum(axis=1) == 1):
            raise ValueError("Macro-F1 in this protocol requires exactly one label per audio clip.")
        precision, recall, f1, _ = precision_recall_fscore_support(
            y.argmax(axis=1), order[:, 0], labels=np.arange(c), zero_division=0)
        result['Macro-F1'] = float(100 * f1.mean())
        per_class.update(precision=(100 * precision).tolist(), recall=(100 * recall).tolist(), f1=(100 * f1).tolist())
    elif task == 'multilabel':
        # No softmax, per-clip score normalization, fitted threshold, or calibration.
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='No positive class found')
            ap = average_precision_score(y, scores, average=None)
        ap = np.asarray(ap).reshape(-1)
        result['mAP'] = float(100 * ap.mean())
        per_class['AP'] = (100 * ap).tolist()
    else:
        raise ValueError("task must be single-label or multilabel.")
    return result, per_class, ranks
