from saki.paths import release_path
"""Resume-safe generation of only the missing exact-head AAKV triples."""
import argparse
import importlib.util
import json
import os
from collections import Counter
from pathlib import Path
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
ROOT = Path(release_path('${RUN_ROOT}'))
OUT = ROOT / '31_corrected_aakv_cache'
MODEL = Path(release_path('${MODEL_ROOT}/千问'))
spec = importlib.util.spec_from_file_location('qwen_source', ROOT / '10_hard-prefix-pilot/code/run_qwen_verified.py')
q = importlib.util.module_from_spec(spec)
spec.loader.exec_module(q)

def atomic_json(path, obj):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False), encoding='utf-8')
    os.replace(tmp, path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=['dcase', 'remaining'], required=True)
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--num-shards', type=int, default=1)
    args = parser.parse_args()
    assert 0 <= args.shard < args.num_shards
    missing = json.loads((OUT / 'missing.json').read_text(encoding='utf-8'))
    selected = [x for x in missing if (x['head'] == 'air horn, truck horn') == (args.phase == 'dcase')]
    items = [x for i, x in enumerate(selected) if i % args.num_shards == args.shard]
    path = OUT / f'generated_{args.phase}_{args.shard}_of_{args.num_shards}.json'
    state = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {'next': 0, 'rows': []}
    assert len(state['rows']) == state['next']
    if state['next'] >= len(items):
        print(f'{path}: already complete ({len(items)})')
        return
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map='cuda:0', local_files_only=True)
    model.eval()
    rows = state['rows']
    for i in tqdm(range(state['next'], len(items)), initial=state['next'], total=len(items), desc=f'{args.phase} shard {args.shard}', mininterval=10):
        x = items[i]
        h, r, t = (x['head'], x['relation'], x['tail'])
        messages = [{'role': 'system', 'content': q.SYSTEM_PROMPT.replace('{head}', h)}, {'role': 'user', 'content': f'[head] {h}\n[relation] {r}\n[tail] {t}'}]
        raw = q.model_generate(model, tokenizer, messages)
        clean, reasons = q.validate(h, r, t, raw)
        verified, outcome, repair_reasons = (clean, 'accepted_first', [])
        if reasons:
            repair = q.REPAIR_PROMPT.replace('{head}', h) + '\nFailed checks: ' + ', '.join(reasons)
            repaired = q.model_generate(model, tokenizer, messages + [{'role': 'assistant', 'content': raw}, {'role': 'user', 'content': repair}])
            candidate, repair_reasons = q.validate(h, r, t, repaired)
            if repair_reasons:
                verified = f'The sound of {h} has the relation {r} with {t}.'
                outcome = 'universal_fallback'
            else:
                verified, outcome = (candidate, 'accepted_repair')
        assert verified.startswith('The sound of ' + h), (x, verified)
        rows.append({**x, 'qwen_verified': verified, 'outcome': outcome, 'first_failures': reasons, 'repair_failures': repair_reasons})
        if (i + 1) % 10 == 0:
            atomic_json(path, {'completed': False, 'next': i + 1, 'total': len(items), 'rows': rows})
    atomic_json(path, {'completed': True, 'next': len(items), 'total': len(items), 'rows': rows, 'audit': dict(Counter((x['outcome'] for x in rows)))})
    print(path, len(rows))
if __name__ == '__main__':
    main()
