"""Retrieve top-M RotatE tails before class-tail filtering and deduplication."""
import argparse
from pathlib import Path
import torch
from sonar.data import read_json, write_json
from sonar.retrieval import retained_tails


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--classes', type=Path, required=True)
    p.add_argument('--relations', type=Path, required=True)
    p.add_argument('--class-to-entity', type=Path, required=True)
    p.add_argument('--entity-to-id', type=Path, required=True)
    p.add_argument('--relation-to-id', type=Path, required=True)
    p.add_argument('--kge-model', type=Path, required=True, help='Trusted upstream PyKEEN model checkpoint, with matching ID maps.')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cuda:0')
    args = p.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    classes, relations = read_json(args.classes), read_json(args.relations)
    mapping, entities, relids = read_json(args.class_to_entity), read_json(args.entity_to_id), read_json(args.relation_to_id)
    model = torch.load(args.kge_model, map_location=args.device, weights_only=False)
    model.to(args.device).eval()
    labels = {int(i): name for name, i in entities.items()}
    tails = {r: {} for r in relations}
    queries = [(ci, r, entities[mapping[c]], relids[r]) for ci, c in enumerate(classes)
               if mapping.get(c) in entities for r in relations]
    with torch.inference_mode():
        for start in range(0, len(queries), 64):
            batch = queries[start:start + 64]
            hr = torch.tensor([[h, r] for _, _, h, r in batch], dtype=torch.long, device=args.device)
            ids = torch.topk(model.score_t(hr), k=3, dim=1).indices.cpu().tolist()
            for (ci, r, _, _), chosen in zip(batch, ids):
                retained = retained_tails([labels[index] for index in chosen], classes)
                if retained: tails[r][str(ci)] = [(tail, classes[ci] + ', ' + tail) for tail in retained]
    write_json(args.output, dict(schema_version=1, classes=classes, relations=relations, top_m=3, tails=tails))


if __name__ == '__main__':
    main()
