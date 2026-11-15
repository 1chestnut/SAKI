"""Online inference from raw audio and prepared text embeddings."""
from dataclasses import dataclass
import random
import json
import torch
import torch.nn.functional as F
from .data import resolve_audio, seed_all
from .scoring import apply_selected, hier, joint, relation_order


@dataclass(frozen=True)
class Branch:
    name: str = 'SoNAR'
    text: str = 'aakv'
    fusion: str = 'hier'
    selection: str = 'sample'
    pool_size: int = 47
    alpha: float = 0.3
    nr: int = 5
    seed: int = 42
    top_k: int = 5
    scale: float = 100.0


class PreparedInputs:
    def __init__(self, path):
        self.payload = torch.load(path, map_location='cpu', weights_only=False)
        if self.payload.get('schema_version') != 1:
            raise ValueError("Unsupported prepared-input schema; run code/prepare_inputs.py.")
        self.classes = self.payload['classes']
        self.relations = self.payload['relations']
        self.records = {int(ci): {r: [(t, int(i)) for t, i in items] for r, items in rows.items()}
                        for ci, rows in self.payload['records'].items()}
        self.counts = {r: sum(len(x.get(r, [])) for x in self.records.values()) for r in self.relations}
        self.frequency_order = sorted(self.relations, key=lambda r: (-self.counts[r], r))

    def activate(self, branch, device):
        return ActiveScorer(self, branch, torch.device(device))


class ActiveScorer:
    def __init__(self, prepared, branch, device):
        self.branch, self.prepared, self.device = branch, prepared, device
        p = prepared.payload
        self.labels = p['label_embeddings'].to(device)
        if not 1 <= branch.pool_size <= len(prepared.relations) or not 1 <= branch.nr <= branch.pool_size:
            raise ValueError("Invalid candidate-pool or selected-relation count.")
        membership = set(prepared.relations if branch.pool_size == len(prepared.relations)
                         else prepared.frequency_order[:branch.pool_size])
        self.relations = [r for r in prepared.relations if r in membership]
        self.embeddings = self.fixed_embeddings = None
        self.records, self.fixed_indices = {}, None
        self.cache_mib = self.labels.numel() * self.labels.element_size() / 2**20
        if branch.selection == 'base':
            return
        if branch.selection == 'fixed_texts':
            self.fixed_embeddings = p['fixed_embeddings'].to(device)
            self.fixed_indices = p['fixed_indices']
            self.cache_mib += (self.fixed_embeddings.numel() * self.fixed_embeddings.element_size()
                               + len(json.dumps(self.fixed_indices).encode())) / 2**20
            return
        ids = sorted({idx for ci in prepared.records.values() for r in self.relations
                      for tail, idx in ci.get(r, [])})
        reindex = {old: new for new, old in enumerate(ids)}
        self.embeddings = p['embeddings'][branch.text][ids].to(device)
        self.records = {ci: {r: [(tail, reindex[idx]) for tail, idx in rows.get(r, [])]
                             for r in self.relations} for ci, rows in prepared.records.items()}
        self.cache_mib += (self.embeddings.numel() * self.embeddings.element_size()
                          + len(json.dumps(self.records, ensure_ascii=False, default=str).encode('utf-8'))) / 2**20

    @torch.inference_mode()
    def score(self, audio, selection_seeds=None):
        branch = self.branch
        audio = audio.reshape(1, -1)
        base = (audio @ self.labels.T).squeeze(0)
        top = torch.argsort(base, descending=True)[:branch.top_k].tolist()
        if branch.selection == 'base':
            return base, {}
        if branch.selection == 'fixed_texts':
            out = base.clone()
            for ci in top:
                ids = self.fixed_indices[ci][branch.text]
                evidence = (audio @ self.fixed_embeddings[ids].T).reshape(-1)
                out[ci] = (joint(base[ci], evidence, branch.scale) if branch.fusion == 'joint'
                           else hier(base[ci], evidence, branch.alpha, branch.scale))
            return out, {}
        similarities = (audio @ self.embeddings.T).reshape(-1)
        per = {ci: {r: [(tail, similarities[idx], similarities[idx]) for tail, idx in self.records[ci][r]]
                    for r in self.relations} for ci in top}
        meta = {}
        if branch.selection == 'sample':
            spaces = {r: apply_selected(base, top, per, [r], branch.text, branch.fusion, branch.alpha, branch.scale)
                      for r in self.relations}
            order, consensus, votes = relation_order(spaces, self.relations)
            chosen = order[:branch.nr]
            meta = dict(consensus_class=consensus, consensus_votes=votes)
        elif branch.selection == 'frozen':
            chosen = [r for r in self.prepared.payload['frozen_relations'] if r in self.relations]
        elif branch.selection == 'frequency':
            chosen = [r for r in self.prepared.frequency_order if r in self.relations][:branch.nr]
        elif branch.selection == 'random':
            if selection_seeds is None or str(branch.seed) not in selection_seeds:
                raise ValueError("Use the manifest's stored relation-selection seed; do not hash a relocated path.")
            chosen = sorted(random.Random(int(selection_seeds[str(branch.seed)])).sample(self.relations, branch.nr))
        else:
            raise ValueError("Unknown relation-selection strategy.")
        out = apply_selected(base, top, per, chosen, branch.text, branch.fusion, branch.alpha, branch.scale)
        return out, dict(initial_top_k_classes=top, selected_relations=chosen, **meta)


def load_clap(weights, device):
    from msclap import CLAP
    model = CLAP(model_fp=str(weights), version='2023', use_cuda=str(device).startswith('cuda'))
    model.clap.eval()
    return model


@torch.inference_mode()
def encode_audio(model, row, audio_root, device):
    # CLAP performs decoding, 44.1-kHz resampling and seven-second repeat/crop.
    # Preserve the accepted seed, independent of the location of the downloaded audio.
    seed_all(row['seed'])
    path = resolve_audio(row, audio_root)
    if not path.is_file():
        raise FileNotFoundError(path)
    embedding = torch.as_tensor(model.get_audio_embeddings([str(path)]), device=device, dtype=torch.float32)
    return F.normalize(embedding, dim=-1)
