"""Export retained graph evidence as inputs to offline AAKV generation."""
import argparse
from pathlib import Path
from sonar.data import read_json, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--graph', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    graph = read_json(args.graph)
    triples = {}
    for relation in graph['relations']:
        for ci, items in graph['tails'][relation].items():
            head = graph['classes'][int(ci)]
            for tail, _ in items:
                triples.setdefault((head, relation, tail), dict(head=head, relation=relation, tail=tail))
    write_json(args.output, list(triples.values()))


if __name__ == '__main__':
    main()
