from pathlib import Path
import argparse
import csv
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument('--baseline', action='store_true')
args = parser.parse_args()
with (ROOT / 'source_prediction_transitions.csv').open(encoding='utf-8-sig', newline='') as f:
    rows = list(csv.DictReader(f))
assert len(rows) == 5
datasets = [row['dataset'] for row in rows]
rescued = np.array([int(row['wrong_to_correct']) for row in rows])
harmed = np.array([int(row['correct_to_wrong']) for row in rows])
assert np.array_equal(rescued-harmed, [int(row['net_rescued']) for row in rows])

mpl.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'mathtext.fontset': 'stix',
    'font.size': 7, 'axes.labelsize': 7, 'axes.titlesize': 8,
    'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5, 'legend.fontsize': 6.5,
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': 0.7,
    'svg.fonttype': 'none', 'pdf.fonttype': 42,
})
fig, ax = plt.subplots(figsize=(3.35, 3.30) if args.baseline else (3.60, 3.50))
y = np.arange(len(rows)) if args.baseline else np.arange(len(rows))*1.15
height = 0.30 if args.baseline else 0.28
b1 = ax.barh(y-height/2, rescued, height, color='#79A1C2', label=r'Wrong $\rightarrow$ correct')
b2 = ax.barh(y+height/2, harmed, height, color='#BFC557', label=r'Correct $\rightarrow$ wrong')
for bars in (b1, b2):
    for b in bars:
        ax.text(b.get_width()+18, b.get_y()+b.get_height()/2,
                f'{int(b.get_width()):,}', ha='left', va='center', fontsize=8.2)
ax.set_yticks(y, datasets)
ax.invert_yaxis()
ax.set_xlabel('Number of test samples')
ax.xaxis.label.set_size(8.2)
ax.tick_params(axis='both', labelsize=8.2)
ax.set_xlim(0, 1340)
ax.grid(False)
ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.01), ncol=2,
          fontsize=8.3, frameon=False, columnspacing=0.7, handlelength=1.5)
if args.baseline:
    fig.subplots_adjust(top=0.86)
    out = ROOT / 'qa' / 'figure3_baseline'
else:
    # Dedicated left label column; dataset names stay unabridged and horizontal.
    ax.set_xticks(np.arange(0, 1201, 200))
    ax.tick_params(axis='y', labelsize=9.0, pad=8, length=0)
    fig.subplots_adjust(left=0.30, right=0.96, bottom=0.14, top=0.86)
    out = ROOT / 'fig_prediction_transitions'
out.parent.mkdir(exist_ok=True)
fig.savefig(out.with_suffix('.pdf'), bbox_inches='tight', facecolor='white')
fig.savefig(out.with_suffix('.svg'), bbox_inches='tight', facecolor='white')
fig.savefig(out.with_suffix('.png'), dpi=600, bbox_inches='tight', facecolor='white')
if not args.baseline:
    fig.savefig(out.with_suffix('.tiff'), dpi=600, bbox_inches='tight', facecolor='white')
plt.close(fig)
print(out)
