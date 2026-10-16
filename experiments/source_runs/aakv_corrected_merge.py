from saki.paths import release_path
"""Verify and merge generated exact-head texts without touching prior caches."""
import argparse
import json
from collections import Counter
from pathlib import Path
ROOT = Path(release_path('${RUN_ROOT}'))
OUT = ROOT / '31_corrected_aakv_cache'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=['dcase', 'all'], required=True)
    args = parser.parse_args()
    missing = json.loads((OUT / 'missing.json').read_text(encoding='utf-8'))
    reused = json.loads((OUT / 'reused_texts.json').read_text(encoding='utf-8'))
    generated = {}
    audit = Counter()
    paths = sorted(OUT.glob('generated_*.json'))
    for path in paths:
        state = json.loads(path.read_text(encoding='utf-8'))
        if not state['completed']:
            if args.phase == 'all':
                raise RuntimeError(f'Unfinished shard: {path}')
            continue
        for row in state['rows']:
            k = '\t'.join((row['head'], row['relation'], row['tail']))
            if k in generated:
                assert generated[k] == row['qwen_verified'], k
            generated[k] = row['qwen_verified']
            audit[row['outcome']] += 1
    required_dcase = {'\t'.join((x['head'], x['relation'], x['tail'])) for x in missing if x['head'] == 'air horn, truck horn'}
    assert required_dcase <= generated.keys(), f'DCASE missing {len(required_dcase - generated.keys())}'
    target = set(reused) | (required_dcase if args.phase == 'dcase' else {'\t'.join((x['head'], x['relation'], x['tail'])) for x in missing})
    texts = dict(reused)
    for k in target - texts.keys():
        assert k in generated, f'Missing generated triple: {k}'
        texts[k] = generated[k]
    dest = OUT / ('aakv_dcase.json' if args.phase == 'dcase' else 'aakv_all.json')
    dest.write_text(json.dumps({'texts': texts}, ensure_ascii=False), encoding='utf-8')
    report = {'phase': args.phase, 'text_keys': len(texts), 'generated_keys_available': len(generated), 'generated_outcomes': dict(audit), 'source_files': [p.name for p in paths]}
    (OUT / f'merge_{args.phase}_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))
if __name__ == '__main__':
    main()
