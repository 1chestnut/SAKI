from saki.paths import release_path
"""CPU-only audit of final sample-level relation choices."""
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
ROOT = Path(release_path('${RUN_ROOT}'))
OUT = ROOT / '31_corrected_aakv_cache/selector_behavior'
DATASETS = [('ESC-50', '01_ESC50', 2000), ('UrbanSound8K', '02_UrbanSound8K', 8732), ('FSD50K', '03_FSD50K', 10231), ('AudioSet', '05_AudioSet', 17233), ('TUT2017', '06_TUT2017', 6300)]

def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def check_five(relations, context):
    if not isinstance(relations, list) or len(relations) != 5 or len(set(relations)) != 5:
        raise ValueError(f'{context}: expected five distinct relations')
    if not all((isinstance(item, str) and item for item in relations)):
        raise ValueError(f'{context}: malformed relation name')
    return tuple(relations)

def check_frozen(relations, context):
    if not isinstance(relations, list) or not relations or len(set(relations)) != len(relations):
        raise ValueError(f'{context}: expected a nonempty fixed set of distinct relations')
    if not all((isinstance(item, str) and item for item in relations)):
        raise ValueError(f'{context}: malformed relation name')
    return tuple(relations)

def sample_ids(rows, context):
    ids = [row['sample_index'] for row in rows]
    if len(set(ids)) != len(ids) or set(ids) != set(range(len(ids))):
        raise ValueError(f'{context}: duplicate or missing sample_index')
    return ids

def write_csv(path, fieldnames, rows):
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

def main():
    summary = []
    frequencies = []
    details = []
    all_relations = set()
    sources = []
    for dataset, code, expected_n in DATASETS:
        if dataset == 'TUT2017':
            primary = ROOT / '28_TUT2017_full6300_rerun/selector' / code / 'samples.json'
            check = ROOT / '28_TUT2017_full6300_rerun/main/test' / code / 'samples.json'
        elif code in ('03_FSD50K', '05_AudioSet'):
            primary = ROOT / '31_corrected_aakv_cache/selector_control' / code / 'samples.json'
            check = ROOT / '31_corrected_aakv_cache/factorial/test' / code / 'samples.json'
        else:
            primary = ROOT / '26_relation_selection_AAKV_control' / code / 'samples.json'
            check = ROOT / '25_final_experiment_alpha03_Nr5/test' / code / 'samples.json'
        rows = read_json(primary)
        check_rows = read_json(check)
        if len(rows) != expected_n or len(check_rows) != expected_n:
            raise ValueError(f'{dataset}: source counts {len(rows)}, {len(check_rows)}; expected {expected_n}')
        sample_ids(rows, str(primary))
        sample_ids(check_rows, str(check))
        check_by_id = {row['sample_index']: row for row in check_rows}
        sets = Counter()
        frequency = Counter()
        jaccard_sum = 0.0
        frozen_reference = None
        for row in rows:
            sample_index = row['sample_index']
            chosen = check_five(row['relations']['AAKV-Selector5'], f'{dataset} #{sample_index} selector')
            frozen = check_frozen(row['relations']['AAKV-FrozenRq'], f'{dataset} #{sample_index} FrozenRq')
            if frozen_reference is None:
                frozen_reference = frozenset(frozen)
            if frozenset(frozen) != frozen_reference:
                raise ValueError(f'{dataset}: FrozenRq differs between samples')
            peer = check_by_id[sample_index]
            if row['audio_path'] != peer['audio_path']:
                raise ValueError(f'{dataset} #{sample_index}: audio path mismatch with final result')
            if dataset == 'TUT2017':
                peer_chosen = check_five(peer['selected']['TFS_Final']['relations'], f'{dataset} #{sample_index} final')
            else:
                peer_chosen = check_five(peer['selected']['TFS_Final']['relations'], f'{dataset} #{sample_index} final')
            if chosen != peer_chosen:
                raise ValueError(f'{dataset} #{sample_index}: selector differs from final prediction')
            choice_set = frozenset(chosen)
            sets[tuple(sorted(choice_set))] += 1
            frequency.update(choice_set)
            jaccard_sum += len(choice_set & frozen_reference) / len(choice_set | frozen_reference)
            all_relations.update(chosen)
            all_relations.update(frozen)
        dominant_set, dominant_count = sets.most_common(1)[0]
        details.append({'dataset': dataset, 'n': expected_n, 'frozen': sorted(frozen_reference), 'dominant': list(dominant_set), 'dominant_count': dominant_count, 'frequency': frequency})
        summary.append({'Dataset': dataset, 'N': expected_n, 'Unique relation sets': len(sets), 'Dominant-set share (%)': f'{100 * dominant_count / expected_n:.6f}', 'Mean Jaccard vs FrozenRq': f'{jaccard_sum / expected_n:.6f}'})
        sources.extend([(primary, digest(primary)), (check, digest(check))])
    if len(all_relations) != 47:
        raise ValueError(f'Observed relation vocabulary has {len(all_relations)} names, not 47; obtain canonical pool before writing CSV')
    for detail in details:
        for relation in sorted(all_relations):
            count = detail['frequency'][relation]
            frequencies.append({'Dataset': detail['dataset'], 'Relation': relation, 'Selected count': count, 'Selected share (%)': f"{100 * count / detail['n']:.6f}", 'In FrozenRq': 'yes' if relation in detail['frozen'] else 'no'})
        if sum(detail['frequency'].values()) != 5 * detail['n']:
            raise ValueError(f"{detail['dataset']}: frequency total does not equal 5N")
    OUT.mkdir(parents=True, exist_ok=True)
    write_csv(OUT / 'selector_behavior_summary.csv', list(summary[0]), summary)
    write_csv(OUT / 'relation_frequency.csv', list(frequencies[0]), frequencies)
    lines = ['# Final sample-level relation selection audit', '', 'This is a CPU-only recount of the completed task 31 and unchanged per-sample selections.', '', '## Sources and checks', '', '- FSD50K and AudioSet use task 31 corrected selector and factorial records; ESC-50 and UrbanSound8K use unchanged task 26 and task 25 records. Selections are cross-checked sample by sample.', '- TUT2017: `28_TUT2017_full6300_rerun/selector/06_TUT2017/samples.json` (`relations.AAKV-Selector5` and `relations.AAKV-FrozenRq`), cross-checked against the 28 final main/test record. The old 4,680-sample Task25e output was not used.', '- All five datasets have the requested N, consecutive unique sample indices, matching final audio paths and relation lists, five distinct selected relations per sample, and one fixed FrozenRq set per dataset.', '- FrozenRq set sizes are dataset-specific: five for ESC-50 and TUT2017, six for UrbanSound8K, and four for FSD50K and AudioSet. Jaccard uses each full fixed set without truncation.', "- The observed vocabulary contains 47 distinct relations. The frequency file contains all 47 for each dataset, including any with zero selections; each dataset's selected counts sum to 5N.", '', '## Definitions', '', '- **Unique relation sets**: number of distinct unordered Top-5 relation combinations; selection order is ignored.', '- **Dominant-set share (%)**: 100 times the count of the most frequent unordered Top-5 combination divided by N.', '- **Selected count/share (%)**: number/percentage of clips whose Top-5 contains a given relation. Shares over 47 relations sum to 500%, because each clip contributes five selections.', '- **Mean Jaccard vs FrozenRq**: arithmetic mean across clips of |selected Top-5 intersection FrozenRq| / |selected Top-5 union FrozenRq|.', '- These measures establish whether selections vary across clips; they do not by themselves establish that selection causes higher accuracy.', '', '## Dominant sets', '', '| Dataset | Count | Relations (unordered) | FrozenRq |', '|---|---:|---|---|']
    for detail in details:
        lines.append(f"| {detail['dataset']} | {detail['dominant_count']} | {', '.join(detail['dominant'])} | {', '.join(detail['frozen'])} |")
    lines += ['', '## Source SHA-256', '']
    for path, sha in sources:
        lines.append(f'- `{path}`: `{sha}`')
    (OUT / 'README.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps({'summary': summary, 'relation_frequency_rows': len(frequencies), 'output': str(OUT)}, ensure_ascii=False))
if __name__ == '__main__':
    main()
