"""Scoring functions extracted verbatim from the corrected paper implementation.
Defaults are the paper setting; never load target labels during selection.
"""
import math
from collections import Counter, OrderedDict
import numpy as np
import torch

def nlse(v):return (torch.logsumexp(100.*v,0)-math.log(v.numel()))/100.

def joint(b,e):return b if not e.numel() else nlse(torch.cat((b.reshape(1),e.reshape(-1))))

def hier(b,e):return b if not e.numel() else .3*b+.7*nlse(e.reshape(-1))

def rank(z,t):
 o=torch.argsort(z,descending=True).cpu().numpy();return min(int(np.where(o==int(y))[0][0])+1 for y in t)

def relation_order(spaces,relations):
 pred={};margin={}
 for r in relations:
  v=torch.topk(spaces[r],k=min(2,spaces[r].numel())).values
  pred[r]=int(torch.argmax(spaces[r]));margin[r]=float(v[0]-v[1]) if v.numel()>1 else 0.
 votes=Counter(pred.values());mv=max(votes.values());cand=sorted(c for c,n in votes.items() if n==mv)
 cc=max(cand,key=lambda c:(sum(margin[r] for r in relations if pred[r]==c),-c))
 mo=sorted(relations,key=lambda r:(-margin[r],r));support=sorted((r for r in relations if pred[r]==cc),key=lambda r:(-margin[r],r))
 return support+[r for r in mo if r not in support],cc,mv

def apply_selected(base,top,per_class,chosen,text_mode,fusion):
 """Construct from immutable base; never consumes another method's scores."""
 out=base.clone()
 for ci in top:
  merged=OrderedDict()
  for r in chosen:
   for tail,direct,aakv in per_class[ci][r]:merged.setdefault(tail,direct if text_mode=='direct' else aakv)
  vals=list(merged.values());ev=torch.stack(vals) if vals else torch.empty(0,device=base.device)
  out[ci]=joint(base[ci],ev) if fusion=='joint' else hier(base[ci],ev)
 return out
