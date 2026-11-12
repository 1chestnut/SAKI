"""Encode the published final method inputs once, outside online timing."""
import argparse
from pathlib import Path
import hashlib
import torch
import torch.nn.functional as F
from saki.data import DATASETS, configure, read_json
from saki.retrieval import encode_texts
from saki.runtime import load_clap


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', choices=DATASETS, required=True)
    p.add_argument('--weights', type=Path, required=True)
    p.add_argument('--input-dir', type=Path, help='Defaults to the published final graph/text inputs.')
    p.add_argument('--aakv', type=Path, help='Optional regenerated AAKV texts; changes the input to the experiment.')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cuda:0')
    args = p.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    configure()
    source = args.input_dir or Path(__file__).resolve().parents[1] / 'inputs' / args.dataset
    graph, fixed = read_json(source / 'graph.json'), read_json(source / 'fixed_evidence.json')
    text_path = args.aakv or source / 'aakv.json'
    texts = read_json(text_path)['texts']
    classes, relations = graph['classes'], graph['relations']
    key_index, keys, records, direct, raw = {}, [], {}, [], []
    for relation in relations:
        for ci_string, items in graph['tails'][relation].items():
            ci = int(ci_string)
            records.setdefault(ci, {})[relation] = []
            for tail, prompt in items:
                key = '\t'.join((classes[ci], relation, tail))
                if key not in texts: raise KeyError('Missing exact class/relation/tail AAKV key: ' + key)
                if key not in key_index:
                    key_index[key] = len(keys); keys.append(key)
                    direct.append(prompt); raw.append(f'{classes[ci]} {relation} {tail}')
                records[ci][relation].append((tail, key_index[key]))
    records = {ci: {r: records.get(ci, {}).get(r, []) for r in relations} for ci in range(len(classes))}
    model = load_clap(args.weights, args.device)
    with torch.inference_mode():
        labels = encode_texts(model, classes, args.device, batch_size=1)
        embeddings = {kind: encode_texts(model, values, args.device, batch_size=128)
                      for kind, values in [('direct', direct), ('raw', raw), ('aakv', [texts[k] for k in keys])]}
        frozen = encode_texts(model, fixed['texts'], args.device, batch_size=1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(schema_version=1, dataset=args.dataset, classes=classes, relations=relations,
                    label_embeddings=labels, embeddings=embeddings, records=records,
                    fixed_embeddings=frozen, fixed_indices=fixed['indices'], frozen_relations=fixed['relations'],
                    source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in (source / 'graph.json', source / 'fixed_evidence.json', text_path)}), args.output)
    print(f'Saved prepared inputs to {args.output}; graph retrieval and text encoding are offline.')


if __name__ == '__main__':
    main()
