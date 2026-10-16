"""Portable paper-setting inference from exact cached resources and raw audio.

This adapter does not replace the preserved executed source runs. Full GPU
end-to-end validation on a new machine remains a release gate.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse
import json
import random
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from msclap import CLAP
from saki.inference import score

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--artifacts',type=Path,required=True)
    parser.add_argument('--dataset',required=True)
    parser.add_argument('--audio-root',type=Path,required=True,help='Root corresponding to DATA_ROOT in the published manifest')
    parser.add_argument('--weights',type=Path,required=True,help='MS-CLAP 2023 model checkpoint')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    parser.add_argument('--limit',type=int,help='Optional smoke-test prefix, not a complete evaluation')
    args=parser.parse_args();root=args.artifacts;ds=args.dataset;device=torch.device(args.device)
    if args.output.exists():raise FileExistsError('Choose a new output; published results are never overwritten.')
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    # Only use trusted, locally obtained project caches. PyTorch 2.4 is the captured version.
    cache=torch.load(root/'resources'/ds/'static_relation_cache.pt',map_location='cpu',weights_only=False)
    frozen=torch.load(root/'resources'/ds/'text_embeddings.pt',map_location='cpu',weights_only=False)
    manifest=json.loads((root/'manifests'/ds/'samples.json').read_text(encoding='utf-8'))
    texts=json.loads((root/'resources/aakv/aakv_all.json').read_text(encoding='utf-8'))['texts']
    model=CLAP(model_fp=str(args.weights),version='2023',use_cuda=args.device=='cuda');model.clap.eval()
    records={r:{} for r in cache['relations']};keys=[];key_index={}
    for rel in cache['relations']:
        for ci,items in cache['tails'][rel].items():
            ci=int(ci);records[rel][ci]=[]
            for tail,prompt in items:
                key='\t'.join((cache['labels'][ci],rel,tail))
                if key not in texts:raise KeyError('Missing exact-label AAKV key: '+key)
                if key not in key_index:key_index[key]=len(keys);keys.append(key)
                records[rel][ci].append((tail,key_index[key]))
    with torch.inference_mode():
        embeddings=[]
        for key in keys:
            emb=torch.as_tensor(model.get_text_embeddings([texts[key]]),device=device,dtype=torch.float32)
            embeddings.append(F.normalize(emb,dim=-1))
        aakv=torch.cat(embeddings);labels=frozen['labels'].to(device)
        rows=[]
        for row in manifest[:args.limit] if args.limit is not None else manifest:
            stored=str(row['audio_path']);prefix='${DATA_ROOT}/'
            if not stored.startswith(prefix):raise ValueError('Unexpected manifest path; do not infer a new sample identity.')
            path=args.audio_root/stored[len(prefix):]
            if not path.is_file():raise FileNotFoundError(path)
            # Preserve original crop seed. Rehashing the portable path would change the audio segment.
            seed=int(row['seed']);random.seed(seed);np.random.seed(seed%(2**32-1))
            torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
            audio=F.normalize(torch.as_tensor(model.get_audio_embeddings([str(path)]),device=device,dtype=torch.float32),dim=-1)
            base=(audio@labels.T).squeeze(0);similarity=(audio@aakv.T).squeeze(0)
            top=torch.argsort(base,descending=True)[:5].tolist()
            per={ci:{rel:[(tail,similarity[index],similarity[index]) for tail,index in records[rel].get(ci,[])]
                     for rel in cache['relations']} for ci in top}
            scores,meta=score(base,per,cache['relations'])
            order=torch.argsort(scores,descending=True).tolist()
            rows.append(dict(sample_index=row['sample_index'],predicted_class=cache['labels'][order[0]],
                             predicted_index=order[0],class_order=order,scores=scores.cpu().tolist(),**meta))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps({'dataset':ds,'scope':'portable adapter; compare against frozen outputs',
                                     'N':len(rows),'results':rows},indent=2)+'\n',encoding='utf-8')
    print('Saved',len(rows),'predictions to',args.output)

if __name__=='__main__':main()
