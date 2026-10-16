"""Exact all-relation retrieval/filtering/cache functions used by the executed pipeline.
The historical analysis around these functions is not included.
"""
from collections import Counter, OrderedDict
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

def build_static_cache(base, clap, kge, factory, labels, mapping, relations,
                       top_m, device, cache_path):
    if cache_path.exists():
        cache = torch.load(cache_path, map_location="cpu")
        if cache.get("relations") == relations and cache.get("labels") == labels:
            return cache

    label_set = {str(x).lower() for x in labels}
    tails = {r: {} for r in relations}
    query_audit = Counter()
    unique_prompts = OrderedDict()

    queries = []
    for ci, label in enumerate(labels):
        entry = mapping[str(label)]
        head = entry["entity"]
        if head not in factory.entity_to_id:
            query_audit["unmapped_classes"] += 1
            continue
        for relation in relations:
            queries.append((ci, label, relation,
                            factory.entity_to_id[head],
                            factory.relation_to_id[relation]))

    entity_labels = {v: k for k, v in factory.entity_to_id.items()}
    predicted_by_query = {}
    query_batch_size = 64
    for start in tqdm(range(0, len(queries), query_batch_size),
                      desc="batched KG relation queries"):
        chunk = queries[start:start + query_batch_size]
        hr = torch.tensor([[x[3], x[4]] for x in chunk], dtype=torch.long,
                          device=device)
        with torch.inference_mode():
            scores = kge.score_t(hr)
            ids = torch.topk(scores, k=top_m, dim=1).indices.cpu().tolist()
        for query, tail_ids in zip(chunk, ids):
            predicted_by_query[(query[0], query[2])] = [
                entity_labels[x] for x in tail_ids]

    for ci, label, relation, _, _ in queries:
        predicted = predicted_by_query[(ci, relation)]
        query_audit["nonempty_queries" if predicted else "empty_queries"] += 1
        kept = OrderedDict()
        for tail in predicted:
            norm = str(tail).lower().strip()
            if norm == str(label).lower() or norm in label_set:
                query_audit["filtered_candidate_tails"] += 1
                continue
            if norm not in kept:
                prompt = f"{label}, {tail}"
                kept[norm] = prompt
                unique_prompts.setdefault(prompt, len(unique_prompts))
        if kept:
            tails[relation][ci] = list(kept.items())

    prompts = list(unique_prompts)
    embeddings = encode_texts(clap, prompts, device)
    cache = {
        "relations": relations,
        "labels": labels,
        "top_m": top_m,
        "tails": tails,
        "prompts": prompts,
        "prompt_embeddings": embeddings,
        "query_audit": dict(query_audit),
    }
    torch.save(cache, cache_path)
    return cache
