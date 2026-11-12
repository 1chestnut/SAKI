"""Check final manifests, evidence inputs, exact keys and optional local audio."""
import argparse
from pathlib import Path
from saki.data import DATASETS, load_manifest, read_json, resolve_audio


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audio-root', type=Path)
    args = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    for ds, (name, task, n, c) in DATASETS.items():
        classes = read_json(root / 'manifests' / ds / 'classes.json')
        relations = read_json(root / 'manifests' / ds / 'relations.json')
        rows = load_manifest(root / 'manifests' / ds / 'samples.json', classes)
        assert len(rows) == n and len(classes) == c and len(relations) == 47
        graph = read_json(root / 'inputs' / ds / 'graph.json')
        mapping = read_json(root / 'inputs' / ds / 'class_to_entity.json')
        texts = read_json(root / 'inputs' / ds / 'aakv.json')['texts']
        assert graph['classes'] == classes and graph['relations'] == relations
        for relation, evidence in graph['tails'].items():
            for ci, items in evidence.items():
                assert not items or mapping[classes[int(ci)]] is not None
                assert len(items) <= 3 and len({x[0] for x in items}) == len(items)
                for tail, direct in items:
                    assert '\t'.join((classes[int(ci)], relation, tail)) in texts
                    assert direct == classes[int(ci)] + ', ' + tail
        assert all(set(r['relation_selection_seeds']) == {'42', '43', '44'} for r in rows)
        if task == 'single-label': assert all(len(r['true_indices']) == 1 for r in rows)
        if args.audio_root:
            missing = [str(resolve_audio(r, args.audio_root)) for r in rows if not resolve_audio(r, args.audio_root).is_file()]
            if missing: raise FileNotFoundError(f'{ds}: {len(missing)} missing files; first: {missing[0]}')
        print(f'{name}: {n} clips, {c} classes, 47 relations; checks passed.')
    dev = load_manifest(root / 'manifests/04_DCASE17_T4/samples.json')
    audioset = load_manifest(root / 'manifests/05_AudioSet/samples.json')
    import re
    dev_ids = {re.match(r'^Y(.{11})_', Path(x['audio_path']).name).group(1) for x in dev}
    test_ids = {Path(x['audio_path']).stem for x in audioset}
    assert not dev_ids.intersection(test_ids), 'Development/test source overlap.'


if __name__ == '__main__':
    main()
