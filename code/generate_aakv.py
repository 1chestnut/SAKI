"""Generate constrained descriptions with the exact paper prompts and repair rule."""
import argparse
import json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from saki.aakv import SYSTEM_PROMPT,REPAIR_PROMPT,model_generate,validate

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--triples',type=Path,required=True,help='JSON list with head, relation and tail fields')
    parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Choose a new output; frozen paper texts must remain unchanged.')
    tokenizer=AutoTokenizer.from_pretrained(args.model,local_files_only=True)
    model=AutoModelForCausalLM.from_pretrained(args.model,torch_dtype=torch.bfloat16,device_map='cuda:0',local_files_only=True)
    model.eval();rows=[];texts={}
    for item in json.loads(args.triples.read_text(encoding='utf-8')):
        h,r,t=(item[k] for k in ['head','relation','tail'])
        messages=[{'role':'system','content':SYSTEM_PROMPT.replace('{head}',h)},
                  {'role':'user','content':f'[head] {h}\n[relation] {r}\n[tail] {t}'}]
        raw=model_generate(model,tokenizer,messages);text,reasons=validate(h,r,t,raw)
        state='accepted_first';repair_reasons=[]
        if reasons:
            repair=REPAIR_PROMPT.replace('{head}',h)+'\nFailed checks: '+', '.join(reasons)
            generated=model_generate(model,tokenizer,messages+[{'role':'assistant','content':raw},{'role':'user','content':repair}])
            text,repair_reasons=validate(h,r,t,generated)
            state='accepted_repair'
            if repair_reasons:text=f'The sound of {h} has the relation {r} with {t}.';state='universal_fallback'
        key='\t'.join((h,r,t))
        if key in texts and texts[key]!=text:raise ValueError('Conflicting duplicate triple')
        texts[key]=text;rows.append(dict(item,first_failures=reasons,repair_failures=repair_reasons,outcome=state,text=text))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps({'texts':texts,'audit':rows},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

if __name__=='__main__':main()
