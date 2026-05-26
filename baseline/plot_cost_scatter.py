#!/usr/bin/env python3
"""
ExpertRate vs AttemptRate scatter, Qwen+LLaMA combined per domain.
Solid = Qwen, hollow = LLaMA. Only PBDD labeled.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 8.5,
    'axes.linewidth': 0.7,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 300,
    'savefig.dpi': 300,
})

# Academic palette (colorbrewer-inspired, print-safe)
C = {
    'post': '#D6604D',   # red
    'pre':  '#4393C3',   # blue
    'two':  '#878787',   # gray
    'pbdd': '#1A1A1A',   # near-black
}

DATA = {
    'Qwen-Code': {
        'Self-REF':      (67.86, 100.0, 'post'),
        'Answer Prob':   (68.11, 100.0, 'post'),
        'Post Linear':   (78.67, 100.0, 'post'),
        'Prompt Router': (76.28, 100.0, 'post'),
        'MC Two-Stage':  (78.67,  64.89,'two'),
        'Pre Linear':    (79.18,  20.82,'pre'),
        'RoBERTa':       (90.31,   9.69,'pre'),
        'PBDD':          (68.37,  36.60,'pbdd'),
    },
    'Qwen-Math': {
        'Self-REF':      (47.40, 100.0, 'post'),
        'Answer Prob':   (81.70, 100.0, 'post'),
        'Post Linear':   (46.87, 100.0, 'post'),
        'Prompt Router': (79.58, 100.0, 'post'),
        'MC Two-Stage':  (49.69,  55.87,'two'),
        'Pre Linear':    (51.75,  48.25,'pre'),
        'RoBERTa':       (67.52,  32.48,'pre'),
        'PBDD':          (40.62,  76.20,'pbdd'),
    },
    'Qwen-MMLU': {
        'Self-REF':      (63.54, 100.0, 'post'),
        'Answer Prob':   (82.43, 100.0, 'post'),
        'Post Linear':   (58.64, 100.0, 'post'),
        'Prompt Router': (89.30, 100.0, 'post'),
        'MC Two-Stage':  (59.25,  80.06,'two'),
        'Pre Linear':    (70.94,  29.06,'pre'),
        'RoBERTa':       (78.71,  21.29,'pre'),
        'PBDD':          (51.91,  69.40,'pbdd'),
    },
    'LLaMA-Code': {
        'Self-REF':      (80.10, 100.0, 'post'),
        'Answer Prob':   (74.23, 100.0, 'post'),
        'Post Linear':   (80.51, 100.0, 'post'),
        'Prompt Router': (92.86, 100.0, 'post'),
        'MC Two-Stage':  (80.59,  57.83,'two'),
        'Pre Linear':    (86.45,  13.55,'pre'),
        'RoBERTa':       (89.80,   9.69,'pre'),
        'PBDD':          (76.02,  34.00,'pbdd'),
    },
    'LLaMA-Math': {
        'Self-REF':      (85.18, 100.0, 'post'),
        'Answer Prob':   (94.32, 100.0, 'post'),
        'Post Linear':   (82.56, 100.0, 'post'),
        'Prompt Router': (93.74, 100.0, 'post'),
        'MC Two-Stage':  (82.52,  65.78,'two'),
        'Pre Linear':    (85.33,  14.67,'pre'),
        'RoBERTa':       (88.92,  11.08,'pre'),
        'PBDD':          (79.18,  56.60,'pbdd'),
    },
    'LLaMA-MMLU': {
        'Self-REF':      (51.82, 100.0, 'post'),
        'Answer Prob':   (81.90, 100.0, 'post'),
        'Post Linear':   (53.92, 100.0, 'post'),
        'Prompt Router': (79.77, 100.0, 'post'),
        'MC Two-Stage':  (55.19,  73.99,'two'),
        'Pre Linear':    (68.77,  31.23,'pre'),
        'RoBERTa':       (77.81,  22.19,'pre'),
        'PBDD':          (51.87,  97.80,'pbdd'),
    },
}

alpha = 0.1
DOMAINS = ['Code', 'Math', 'MMLU']

# marker: Qwen=filled circle/square/etc, LLaMA=same but hollow
MARKERS = {'post': 'o', 'pre': 's', 'two': 'D', 'pbdd': '*'}
SIZES   = {'post': 35,  'pre': 35,  'two': 35,  'pbdd': 180}


def draw_panel(ax, domain):
    qwen_data  = DATA[f'Qwen-{domain}']
    llama_data = DATA[f'LLaMA-{domain}']

    # iso-cost lines
    t = np.linspace(0, 105, 300)
    for cost in [55, 70, 85, 100]:
        yl = cost - alpha * t
        mask = (yl >= 30) & (yl <= 102) & (t >= 0) & (t <= 102)
        if mask.sum() < 2:
            continue
        ax.plot(t[mask], yl[mask], color='#CCCCCC', lw=0.7, ls='--', zorder=1)
        ax.text(t[mask][-1] + 0.5, yl[mask][-1], f'{cost}',
                fontsize=5.5, color='#BBBBBB', va='center')

    for dataset, filled in [(qwen_data, True), (llama_data, False)]:
        for m, (er, ar, kind) in dataset.items():
            color  = C[kind]
            marker = MARKERS[kind]
            size   = SIZES[kind]
            is_pbdd = kind == 'pbdd'
            model_tag = 'Qwen' if filled else 'LLaMA'

            if filled:
                ax.scatter(ar, er, s=size, c=color, marker=marker,
                           edgecolors='white' if not is_pbdd else color,
                           linewidths=0.5 if not is_pbdd else 1.5,
                           zorder=5 if is_pbdd else 3, alpha=0.9)
            else:
                ax.scatter(ar, er, s=size, facecolors='none', marker=marker,
                           edgecolors=color,
                           linewidths=1.5 if is_pbdd else 0.8,
                           zorder=5 if is_pbdd else 3, alpha=0.9)

            if is_pbdd:
                dy = 3 if filled else -5
                ax.text(ar - 2, er + dy, model_tag,
                        fontsize=7, color=color, fontweight='bold',
                        ha='right', va='bottom' if dy > 0 else 'top', zorder=6)

    ax.set_xlim(-3, 108)
    ax.set_ylim(35, 100)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel('AttemptRate (%)', fontsize=8.5)
    ax.set_title(domain, fontsize=10, fontweight='bold', pad=5)
    ax.tick_params(labelsize=8)
    ax.grid(alpha=0.12, lw=0.35, zorder=0)
    ax.set_axisbelow(True)
    ax.text(1, 37, r'$\swarrow$ lower is better', fontsize=6, color='#AAAAAA')


# Legend
legend_handles = [
    mpatches.Patch(color=C['post'],  label='Post-gen'),
    mpatches.Patch(color=C['pre'],   label='Pre-gen'),
    mpatches.Patch(color=C['two'],   label='Two-stage probe'),
    mpatches.Patch(color=C['pbdd'],  label='PBDD (ours)'),
    mlines.Line2D([], [], color='k', marker='o', ls='None', ms=5,
                  markerfacecolor='k', label='Qwen (filled)'),
    mlines.Line2D([], [], color='k', marker='o', ls='None', ms=5,
                  markerfacecolor='none', markeredgewidth=0.8, label='LLaMA (hollow)'),
    mlines.Line2D([], [], color='#CCCCCC', ls='--', lw=1,
                  label=r'Iso-cost$_{0.1}$'),
]

out_dir = Path("figures/cost_scatter")
out_dir.mkdir(parents=True, exist_ok=True)

fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.5))
for j, (ax, domain) in enumerate(zip(axes, DOMAINS)):
    draw_panel(ax, domain)
    if j == 0:
        ax.set_ylabel('ExpertRate (%)', fontsize=8.5)

fig.legend(handles=legend_handles, loc='lower center',
           bbox_to_anchor=(0.5, -0.07), ncol=7, fontsize=7.5, frameon=False)
fig.suptitle(r'ExpertRate vs.\ AttemptRate at Route@95\%',
             fontsize=10, fontweight='bold', y=1.02)
plt.tight_layout()
fig.savefig(out_dir / 'cost_scatter.pdf', bbox_inches='tight')
fig.savefig(out_dir / 'cost_scatter.png', bbox_inches='tight')
print("Saved cost_scatter.{pdf,png}")
plt.close(fig)
