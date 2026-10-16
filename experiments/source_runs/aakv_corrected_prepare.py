from saki.paths import release_path
"""Build a versioned AAKV cache using exact dataset class labels as heads."""
import hashlib
import json
from collections import Counter
from pathlib import Path
import torch
ROOT = Path(release_path('${RUN_ROOT}'))
OUT = ROOT / '31_corrected_aakv_cache'
DATASETS = ('01_ESC50', '02_UrbanSound8K', '03_FSD50K', '04_DCASE17_T4', '05_AudioSet', '06_TUT2017')

def key(head, relation, tail):
    return '\t'.join((str(head), str(relation), str(tail)))

def main():
    OUT.mkdir(exist_ok=True)
    old = json.loads((ROOT / '20A_topP5_aligned_selector_exploration/aakv_all.json').read_text(encoding='utf-8'))['texts']
    required = {}
    dataset_keys = {}
    for ds in DATASETS:
        cache = torch.load(ROOT / f'18a_relation_oracle_audit/{ds}/results/static_relation_cache.pt', map_location='cpu')
        labels = cache['labels']
        ds_keys = set()
        for relation in cache['relations']:
            for ci, items in cache['tails'][relation].items():
                head = labels[int(ci)]
                for tail, prompt in items:
                    assert prompt.startswith(head + ', '), (ds, ci, head, prompt)
                    k = key(head, relation, tail)
                    required[k] = {'head': head, 'relation': relation, 'tail': tail}
                    ds_keys.add(k)
        dataset_keys[ds] = ds_keys
    texts = {k: old[k] for k in required if k in old}
    source = Counter({'existing_20A_exact_key': len(texts)})
    for ds in DATASETS:
        old_path = ROOT / f'12_shared-abcd-statistics/{ds}/frozen_inputs/aakv.json'
        rows = json.loads(old_path.read_text(encoding='utf-8'))['rows']
        for row in rows:
            k = key(row['head'], row['relation'], row['tail'])
            if k not in required:
                continue
            value = row['qwen_verified']
            if k in texts:
                assert texts[k] == value, (ds, k, texts[k], value)
            else:
                texts[k] = value
                source['existing_frozen_exact_key'] += 1
    missing = [required[k] for k in sorted(required) if k not in texts]
    report = {'required_unique_keys': len(required), 'reused_existing_keys': len(texts), 'missing_unique_keys': len(missing), 'sources': dict(source), 'dataset': {ds: {'required_unique_keys': len(ks), 'missing_unique_keys': len(ks - texts.keys()), 'affected_classes': sorted({required[k]['head'] for k in ks - texts.keys()})} for ds, ks in dataset_keys.items()}, 'upstream_20A_sha256': hashlib.sha256((ROOT / '20A_topP5_aligned_selector_exploration/aakv_all.json').read_bytes()).hexdigest()}
    (OUT / 'prepare_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    (OUT / 'reused_texts.json').write_text(json.dumps(texts, ensure_ascii=False), encoding='utf-8')
    (OUT / 'missing.json').write_text(json.dumps(missing, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))
if __name__ == '__main__':
    main()
