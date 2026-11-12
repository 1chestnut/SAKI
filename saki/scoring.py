"""Paper scoring rules. Class labels are never inputs to relation selection."""
import math
from collections import Counter, OrderedDict
import numpy as np
import torch


def nlse(values, scale=100.0):
    """Normalized scaled log-sum-exp over nonempty knowledge scores."""
    values = values.reshape(-1)
    if not values.numel() or scale <= 0:
        raise ValueError("Aggregation requires nonempty scores and a positive scale.")
    return (torch.logsumexp(scale * values, 0) - math.log(values.numel())) / scale


def joint(base, evidence, scale=100.0):
    return base if not evidence.numel() else nlse(torch.cat((base.reshape(1), evidence.reshape(-1))), scale)


def hier(base, evidence, alpha=0.3, scale=100.0):
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be in [0, 1].")
    return base if not evidence.numel() else alpha * base + (1 - alpha) * nlse(evidence, scale)


def rank(scores, truth):
    order = torch.argsort(scores, descending=True).cpu().numpy()
    return min(int(np.where(order == int(y))[0][0]) + 1 for y in truth)


def relation_order(spaces, relations):
    """Consensus first, then top-2 score gap, then relation-name order."""
    pred, gap = {}, {}
    for relation in relations:
        scores = spaces[relation]
        best = torch.topk(scores, k=min(2, scores.numel())).values
        pred[relation] = int(torch.argmax(scores))
        gap[relation] = float(best[0] - best[1]) if best.numel() > 1 else 0.0
    votes = Counter(pred.values())
    most = max(votes.values())
    tied = sorted(c for c, count in votes.items() if count == most)
    consensus = max(tied, key=lambda c: (
        sum(gap[r] for r in relations if pred[r] == c), -c))
    by_gap = sorted(relations, key=lambda r: (-gap[r], r))
    agreeing = sorted((r for r in relations if pred[r] == consensus), key=lambda r: (-gap[r], r))
    return agreeing + [r for r in by_gap if r not in agreeing], consensus, most


def apply_selected(base, top, per_class, chosen, text_mode='aakv', fusion='hier', alpha=0.3, scale=100.0):
    """Merge in relation-priority order; the first normalized tail occurrence wins."""
    if text_mode not in {'direct', 'aakv', 'raw'} or fusion not in {'hier', 'joint'}:
        raise ValueError("Unknown text representation or fusion rule.")
    result = base.clone()
    for ci in top:
        merged = OrderedDict()
        for relation in chosen:
            for tail, direct_score, text_score in per_class[ci].get(relation, []):
                merged.setdefault(tail, direct_score if text_mode == 'direct' else text_score)
        evidence = torch.stack(list(merged.values())) if merged else base.new_empty(0)
        result[ci] = (joint(base[ci], evidence, scale) if fusion == 'joint'
                      else hier(base[ci], evidence, alpha, scale))
    return result
