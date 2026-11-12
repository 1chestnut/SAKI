"""Sample-level relation selection and final class scoring."""
import torch
from .scoring import apply_selected, relation_order


@torch.inference_mode()
def score(base_scores, evidence_scores, relations, top_k=5, number_of_relations=5,
          alpha=0.3, scale=100.0, text_mode='aakv', fusion='hier'):
    if base_scores.ndim != 1 or base_scores.numel() < 2:
        raise ValueError("Expected scores for at least two candidate classes.")
    if not relations or len(set(relations)) != len(relations):
        raise ValueError("Relations must be nonempty and unique.")
    if not 1 <= top_k <= base_scores.numel() or not 1 <= number_of_relations <= len(relations):
        raise ValueError("Invalid class or relation count.")
    top = torch.argsort(base_scores, descending=True)[:top_k].tolist()
    per = {ci: {r: evidence_scores.get(ci, {}).get(r, []) for r in relations} for ci in top}
    spaces = {r: apply_selected(base_scores, top, per, [r], text_mode, fusion, alpha, scale)
              for r in relations}
    ordered, consensus, votes = relation_order(spaces, relations)
    chosen = ordered[:number_of_relations]
    final = apply_selected(base_scores, top, per, chosen, text_mode, fusion, alpha, scale)
    return final, dict(initial_top_k_classes=top, selected_relations=chosen,
                       consensus_class=consensus, consensus_votes=votes)
