"""CPU-only reconstruction of final manuscript results; never runs a model."""
import argparse, collections, csv, hashlib, json, pathlib
import numpy as np
from scipy.stats import binomtest

def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def dump(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def csvout(p,rows):
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def metrics(r):
 r=np.asarray(r,float);assert np.isfinite(r).all() and (r>=1).all()
 return {'Hit@1':float(100*np.mean(r==1)),'Hit@3':float(100*np.mean(r<=3)),'Hit@5':float(100*np.mean(r<=5)),'MRR':float(100*np.mean(1/r))}
def same_samples(a,b):
 assert len(a)==len(b)
 for x,y in zip(a,b):assert x['audio_path']==y['audio_path'] and x['true_indices']==y['true_indices']
def holm(vals):
 idx=np.argsort(vals);out=np.zeros(len(vals));prev=0
 for rank,j in enumerate(idx):prev=max(prev,min(1.,(len(vals)-rank)*vals[j]));out[j]=prev
 return out.tolist()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--package',type=pathlib.Path,default=pathlib.Path(__file__).resolve().parents[1]);ap.add_argument('--bootstrap',action='store_true');args=ap.parse_args();p=args.package
 reg=read(p/'provenance/datasets.json'); mainrows=[];ablation=[];selector=[];verbal=[];behavior=[];freq=[];stats=[];audit={};issues=[]
 for code,meta in reg.items():
  name=meta['name'];r=read(p/f'results/factorial/{code}/samples.json');s=read(p/f'results/selector/{code}/samples.json');raw=read(p/f'results/raw_triple/{code}/samples.json');same_samples(r,s);same_samples(r,raw);assert len(r)==meta['n']
  manifest=read(p/f'manifests/{code}/samples.json');same_samples(r,manifest)
  assert len(set(x['audio_path'] for x in r))==len(r)
  orig=read(p/f'results/factorial/{code}/metrics.json')
  computed={m:metrics([x['ranks'][m] for x in r]) for m in r[0]['ranks']}
  for m,v in computed.items():
   for k,val in v.items():assert abs(val-orig[m][k])<1e-7,(code,m,k,val,orig[m][k])
   ablation.append({'dataset':name,'method':m,**v})
   if m in ['CLAP','I_iKnow','TFS_Final']:mainrows.append({'dataset':name,'method':m,**v})
  sorig=read(p/f'results/selector/{code}/metrics.json')
  sc={m:metrics([x['ranks'][m] for x in s]) for m in s[0]['ranks']}
  for m,v in sc.items():
   for k,val in v.items():assert abs(val-sorig[m][k])<1e-7,(code,m,k)
   selector.append({'dataset':name,'method':m,**v})
  assert all(x['ranks']['TF']==y['ranks']['AAKV-FrozenRq'] and x['ranks']['TFS_Final']==y['ranks']['AAKV-Selector5'] for x,y in zip(r,s)),(code,'factorial/control mismatch')
  for label,ranks in [('Direct concatenation',[x['ranks']['FS'] for x in r]),('Raw triple',[x['rank'] for x in raw]),('AAKV',[x['ranks']['TFS_Final'] for x in r])]:verbal.append({'dataset':name,'method':label,**metrics(ranks)})
  # Validate saved ranks against all-class score bounds, allowing stored-score ties.
  pred=np.load(p/f'results/factorial/{code}/predictions.npz',allow_pickle=True)
  a=pred['scores'];methods=pred['methods'].tolist();assert a.shape==(len(r),len(methods),meta['n_classes']);assert np.isfinite(a).all()
  failures=0;tie_rows=0
  for j,x in enumerate(r):
   gt=x['true_indices'];assert all(0<=g<meta['n_classes'] for g in gt)
   for k,m in enumerate(methods):
    z=a[j,k];best=z[gt].max();lo=1+int(np.sum(z>best));hi=int(np.sum(z>=best));rr=x['ranks'][m]
    failures+=not(lo<=rr<=hi);tie_rows+=hi>lo
  audit[code]={'N':len(r),'score_shape':list(a.shape),'score_dtype':str(a.dtype),'rank_outside_score_tie_bounds':int(failures),'score_tie_rows':int(tie_rows),'metrics_match':True,'manifests_aligned':True,'TF_equals_FrozenRq_samplewise':True,'final_equals_Selector_samplewise':True}
  if failures:issues.append(f'{code}: {failures} saved ranks outside stored-score bounds')
  frozen=set(read(p/f'results/selector/{code}/protocol.json')['frozen_relations']);rels=read(p/f'manifests/{code}/relations.json');cnt=collections.Counter();sets=collections.Counter();jac=[]
  for x in r:
   selected=x['selected']['TFS_Final']['relations'];assert len(selected)==5 and len(set(selected))==5 and set(selected)<=set(rels)
   sets[tuple(sorted(selected))]+=1;cnt.update(selected);ss=set(selected);jac.append(len(ss&frozen)/len(ss|frozen))
  behavior.append({'dataset':name,'N':len(r),'unique_top5_sets':len(sets),'dominant_set_share':max(sets.values())/len(r),'mean_jaccard_vs_FrozenRq':float(np.mean(jac))})
  for rel in rels:freq.append({'dataset':name,'relation':rel,'count':cnt[rel],'sample_fraction':cnt[rel]/len(r)})
  ra=np.array([x['ranks']['I_iKnow'] for x in r],float);rb=np.array([x['ranks']['TFS_Final'] for x in r],float);ha=ra==1;hb=rb==1;b=int(np.sum(~ha&hb));c=int(np.sum(ha&~hb))
  row={'dataset':name,'N':len(r),'wrong_to_correct':b,'correct_to_wrong':c,'delta_Hit@1_pp':float(100*np.mean(hb.astype(float)-ha)),'delta_MRR_pp':float(100*np.mean(1/rb-1/ra)),'McNemar_exact_p':float(binomtest(b,b+c,.5).pvalue) if b+c else 1.}
  if args.bootstrap:
   seed=20260911+int(hashlib.sha256((name+'|Final-TFS - Frozen-iKnow').encode()).hexdigest()[:8],16);rng=np.random.default_rng(seed);dh=hb.astype(float)-ha;dr=1/rb-1/ra;hs=[];rs=[]
   for start in range(0,10000,250):
    idx=rng.integers(0,len(r),size=(250,len(r)));hs.extend(dh[idx].mean(1)*100);rs.extend(dr[idx].mean(1)*100)
   row.update(dict(zip(['Hit@1_CI95_low','Hit@1_CI95_high','MRR_CI95_low','MRR_CI95_high'],[*np.percentile(hs,[2.5,97.5]),*np.percentile(rs,[2.5,97.5])])))
  stats.append(row);print(name,'verified',len(r),flush=True)
 for row,v in zip(stats,holm([x['McNemar_exact_p'] for x in stats])):row['McNemar_Holm_p']=v
 for rows in [mainrows,ablation,selector,verbal]:
  for m in dict.fromkeys(x['method'] for x in rows):
   part=[x for x in rows if x['method']==m];assert len(part)==5
   rows.append({'dataset':'Five-dataset unweighted mean','method':m,**{k:float(np.mean([x[k] for x in part])) for k in ['Hit@1','Hit@3','Hit@5','MRR']}})
 for name,rows in [('table1_main',mainrows),('table2_ablation',ablation),('table3_selector_all_seeds',selector),('table4_verbalization',verbal),('selector_behavior',behavior),('relation_frequency',freq),('paired_statistics',stats)]:csvout(p/f'rebuilt/{name}.csv',rows)
 dev=read(p/'results/development/samples.json');assert len(dev)==419
 dm={m:metrics([x['ranks'][m] for x in dev]) for m in dev[0]['ranks']};csvout(p/'rebuilt/development_grid.csv',[{'method':m,**v} for m,v in dm.items()])
 cand=[m for m in dm if m.startswith('TFS_') and not m.endswith('Nr47')];win=sorted(cand,key=lambda m:(-dm[m]['Hit@1'],-dm[m]['MRR'],int(m.split('Nr')[-1])))[0];assert win=='TFS_alpha0.3_Nr5'
 dump(p/'rebuilt/verification.json',{'datasets':audit,'development_N':419,'development_selected':win,'bootstrap_recomputed':args.bootstrap,'issues':issues,'scope':'Stored scores/ranks and frozen final outputs; not a fresh model inference run.'})
 if issues:raise SystemExit('Audit exceptions: '+str(issues))
 print('PASS: all final result sample identities, metrics, score bounds, branch equivalences and development choice verified.')
if __name__=='__main__':main()
