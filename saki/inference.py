"""Thin inference interface around the verbatim paper scoring functions."""
import torch
from .scoring import apply_selected, relation_order

@torch.inference_mode()
def score(base_scores, evidence_scores, relations, top_k=5, number_of_relations=5):
    """Return final scores and selection metadata without reading target labels.

    evidence_scores[class_index][relation] is an ordered list of
    (normalized_tail, direct_score, AAKV_score) tuples. The paper setting
    uses AAKV_score, alpha=.3 and normalized LSE scale=100.
    """
    if base_scores.ndim != 1 or base_scores.numel() < 2:
        raise ValueError('Expected a one-dimensional score vector for all classes.')
    if len(set(relations)) != len(relations) or not relations:
        raise ValueError('Relations must be nonempty and unique.')
    if not 1 <= number_of_relations <= len(relations):
        raise ValueError('Invalid selected-relation count.')
    top = torch.argsort(base_scores, descending=True)[:top_k].tolist()
    per = {ci:{r:evidence_scores.get(ci,{}).get(r,[]) for r in relations} for ci in top}
    spaces = {r:apply_selected(base_scores, top, per, [r], 'aakv', 'hier') for r in relations}
    ordered, consensus, votes = relation_order(spaces, relations)
    chosen = ordered[:number_of_relations]
    final = apply_selected(base_scores, top, per, chosen, 'aakv', 'hier')
    return final, dict(top_k_classes=top, selected_relations=chosen,
                       consensus_class=consensus, consensus_votes=votes)
