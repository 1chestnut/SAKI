"""Offline text encoding and retained top-M RotatE evidence."""
from collections import OrderedDict
import torch
import torch.nn.functional as F
from tqdm import tqdm

def encode_texts(clap, texts, device, batch_size=128):
    output = []
    for start in tqdm(range(0, len(texts), batch_size), desc="text embeddings"):
        batch = texts[start:start + batch_size]
        try:
            emb = clap.get_text_embeddings(batch)
            emb = torch.as_tensor(emb, device=device, dtype=torch.float32)
        except Exception:
            pieces = []
            for text in batch:
                pieces.append(torch.as_tensor(
                    clap.get_text_embeddings([text]), device=device,
                    dtype=torch.float32))
            emb = torch.cat(pieces, dim=0)
        output.append(F.normalize(emb, dim=-1).cpu())
    return torch.cat(output, dim=0)

def retained_tails(labels, candidate_classes):
    """Normalize/filter an already retrieved top-M list; never refill removed tails."""
    class_names = {str(c).lower().strip() for c in candidate_classes}
    retained = OrderedDict()
    for label in labels:
        normalized = str(label).lower().strip()
        if normalized not in class_names:
            retained.setdefault(normalized, None)
    return list(retained)
