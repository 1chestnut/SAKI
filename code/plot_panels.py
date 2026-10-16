"""Regenerate statistical and parameter panels from the paper's source values.

Data are unchanged; this is a portable redraw, not a pixel-identical artwork copy.
"""
import argparse,csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def read(path):
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def main():
    p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True)
    data=Path(__file__).resolve().parents[1]/'figures'
    rows=read(data/'source_statistics_31_20261009.csv')
    fig,axes=plt.subplots(1,2,figsize=(9,3.2),layout='constrained')
    for ax,metric,low,high,title in [(axes[0],'delta_Hit1_pp','Hit1_CI_low','Hit1_CI_high','Hit@1'),
                                    (axes[1],'delta_MRR_pp','MRR_CI_low','MRR_CI_high','MRR')]:
        values=np.array([float(x[metric]) for x in rows]);lo=np.array([float(x[low]) for x in rows]);hi=np.array([float(x[high]) for x in rows])
        ax.errorbar(values,np.arange(len(rows)),xerr=[values-lo,hi-values],fmt='o',capsize=3)
        ax.set_yticks(range(len(rows)),[x['dataset'] for x in rows]);ax.invert_yaxis()
        ax.axvline(0,color='gray',linestyle='--');ax.set_xlabel(title+' difference (percentage points)')
    fig.savefig(a.output_dir/'statistical_significance.pdf');plt.close(fig)
    rows=read(data/'source_parameter_sensitivity.csv');alphas=sorted({float(x['alpha']) for x in rows});numbers=sorted({int(x['N_r']) for x in rows})
    fig,axes=plt.subplots(2,1,figsize=(5,6),layout='constrained')
    for ax,metric in zip(axes,['Hit@1','MRR']):
        grid=np.array([[float(next(x[metric] for x in rows if float(x['alpha'])==alpha and int(x['N_r'])==nr)) for nr in numbers] for alpha in alphas])
        im=ax.imshow(grid,aspect='auto');fig.colorbar(im,ax=ax)
        ax.set_xticks(range(len(numbers)),numbers);ax.set_yticks(range(len(alphas)),alphas)
        ax.set_xlabel('Number of selected relations');ax.set_ylabel('Fusion weight');ax.set_title(metric+' (%)')
        for i in range(len(alphas)):
            for j in range(len(numbers)):ax.text(j,i,f'{grid[i,j]:.2f}',ha='center',va='center')
    fig.savefig(a.output_dir/'parameter_sensitivity.pdf');plt.close(fig)

if __name__=='__main__':main()
