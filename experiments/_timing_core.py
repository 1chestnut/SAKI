"""Accepted timing scorer; retained arithmetic and allocation order."""
import math
from collections import Counter, OrderedDict
import torch
K = 5
NR = 5
KAPPA = 100.0
ALPHA = 0.3

def nlse(v): return (torch.logsumexp(KAPPA*v,0)-math.log(v.numel()))/KAPPA

def joint(b,e): return b if not e.numel() else nlse(torch.cat((b.reshape(1),e.reshape(-1))))

def hier(b,e): return b if not e.numel() else ALPHA*b+(1-ALPHA)*nlse(e.reshape(-1))

def relation_order(spaces, relations):
    pred={};margin={}
    for r in relations:
        vals=torch.topk(spaces[r],k=min(2,spaces[r].numel())).values
        pred[r]=int(torch.argmax(spaces[r]));margin[r]=float(vals[0]-vals[1]) if vals.numel()>1 else 0.
    votes=Counter(pred.values());mv=max(votes.values());cand=sorted(c for c,n in votes.items() if n==mv)
    cc=max(cand,key=lambda c:(sum(margin[r] for r in relations if pred[r]==c),-c))
    mo=sorted(relations,key=lambda r:(-margin[r],r));support=sorted((r for r in relations if pred[r]==cc),key=lambda r:(-margin[r],r))
    return support+[r for r in mo if r not in support]

def final_one(base,audio,top,relations,per,ae):
    # Match version-31 factorial scoring: one audio/evidence matrix product.
    similarity=(audio@ae.T).reshape(-1)
    spaces={}
    for r in relations:
        z=base.clone()
        for ci in top:
            ids=[idx for tail,idx in per[ci][r]];ev=audio.new_empty(0) if not ids else torch.stack([similarity[idx] for idx in ids])
            z[ci]=hier(base[ci],ev)
        spaces[r]=z
    chosen=relation_order(spaces,relations)[:NR];z=base.clone()
    for ci in top:
        merged=OrderedDict()
        for r in chosen:
            for tail,idx in per[ci][r]:merged.setdefault(tail,idx)
        ids=list(merged.values());ev=audio.new_empty(0) if not ids else torch.stack([similarity[idx] for idx in ids])
        z[ci]=hier(base[ci],ev)
    return z

def run_method(method,audios,label_emb,frozen_emb,frozen_idx,relations,per_class,aakv_emb):
    outputs=[]
    for audio in audios:
        base=(audio.reshape(1,-1)@label_emb.T).squeeze(0);top=torch.argsort(base,descending=True)[:K].tolist()
        if method=='CLAP': z=base
        elif method=='iKnow_dagger':
            z=base.clone()
            for ci in top:
                ev=(audio.reshape(1,-1)@frozen_emb[frozen_idx[ci]['direct']].T).reshape(-1);z[ci]=joint(base[ci],ev)
        else:z=final_one(base,audio.reshape(1,-1),top,relations,per_class,aakv_emb)
        outputs.append(z)
    return torch.stack(outputs)

def score_and_rank(*args):
    scores=run_method(*args)
    return scores,torch.argsort(scores,dim=1,descending=True)
