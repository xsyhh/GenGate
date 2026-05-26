#!/usr/bin/env python3
"""
Cost decomposition bar chart: ExpertRate vs AttemptRate per method at Route@95%.
Shows why GenGate achieves lower Cost_alpha: it reduces BOTH components simultaneously,
while pre-gen routers only reduce AttemptRate and post-gen routers only reduce ExpertRate.

Data from Table 1 appendix (joint cost tables).
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.08,
})

# Data from appendix joint-cost tables (ExpertRate, AttemptRate at Route@95%)
# Format: {setting: {method: (expert_rate, attempt_rate)}}
DATA = {
    'Qwen-Code': {
        'Self-REF':        (67.86, 100.0),
        'Answer Prob':     (68.11, 100.0),
        'Post Linear':     (78.67, 100.0),
        'Prompt Router':   (76.28, 100.0),
        'MC Two-Stage':    (78.67,  64.89),
        'Pre Linear':      (79.18,  20.82),
        'RoBERTa Router':  (90.31,   9.69),
        'GenGate':         (68.37,  36.60),
    },
    'Qwen-Math': {
        'Self-REF':        (47.40, 100.0),
        'Answer Prob':     (81.70, 100.0),
        'Post Linear':     (46.87, 100.0),
        'Prompt Router':   (79.58, 100.0),
        'MC Two-Stage':    (49.69,  55.87),
        'Pre Linear':      (51.75,  48.25),
        'RoBERTa Router':  (67.52,  32.48),
        'GenGate':         (40.62,  76.20),
    },
    'Qwen-MMLU': {
        'Self-REF':        (63.54, 100.0),
        'Answer Prob':     (82.43, 100.0),
        'Post Linear':     (58.64, 100.0),
        'Prompt Router':   (89.30, 100.0),
        'MC Two-Stage':    (59.25,  80.06),
        'Pre Linear':      (70.94,  29.06),
        'RoBERTa Router':  (78.71,  21.29),
        'GenGate':         (51.91,  69.40),
    },
    'LLaMA-Code': {
        'Self-REF':        (80.10, 100.0),
        'Answer Prob':     (74.23, 100.0),
        'Post Linear':     (80.51, 100.0),
        'Prompt Router':   (92.86, 100.0),
        'MC Two-Stage':    (80.59,  57.83),
        'Pre Linear':      (86.45,  13.55),
        'RoBERTa Router':  (89.80,   9.69),
        'GenGate':         (76.02,  34.00),
    },
    'LLaMA-Math': {
        'Self-REF':        (85.18, 100.0),
        'Answer Prob':     (94.32, 100.0),
        'Post Linear':     (82.56, 100.0),
        'Prompt Router':   (93.74, 100.0),
        'MC Two-Stage':    (82.52,  65.78),
        'Pre Linear':      (85.33,  14.67),
        'RoBERTa Router':  (88.92,  11.08),
        'GenGate':         (79.18,  56.60),
    },
    'LLaMA-MMLU': {
        'Self-REF':        (51.82, 100.0),
        'Answer Prob':     (81.90, 100.0),
        'Post Linear':     (53.92, 100.0),
        'Prompt Router':   (79.77, 100.0),
        'MC Two-Stage':    (55.19,  73.99),
        'Pre Linear':      (68.77,  31.23),
        'RoBERTa Router':  (77.81,  22.19),
        'GenGate':         (51.87,  97.80),
    },
}

alpha = 0.1
out_dir = Path("figures/cost_decomposition")
out_dir.mkdir(parents=True, exist_ok=True)

METHOD_ORDER = [
    'Post Linear', 'Self-REF', 'Answer Prob', 'Prompt Router',
    'MC Two-Stage', 'Pre Linear', 'RoBERTa Router', 'GenGate',
]
SHORT_LABELS = {
    'Post Linear': 'Post\nLinear', 'Self-REF': 'Self-\nREF',
    'Answer Prob': 'Ans\nProb', 'Prompt Router': 'Prompt\nRouter',
    'MC Two-Stage': 'MC Two-\nStage', 'Pre Linear': 'Pre\nLinear',
    'RoBERTa Router': 'RoBERTa', 'GenGate': 'GenGate',
}

# Academic palette: ExpertRate (steel blue), alpha*AttemptRate (light blue), GenGate (dark navy)
C_EXPERT_BASE  = '#4393C3'
C_ATTEMPT_BASE = '#92C5DE'
C_EXPERT_GENGATE  = '#2166AC'
C_ATTEMPT_GENGATE = '#4393C3'


def draw_panel(ax, setting):
    data = DATA[setting]
    methods = METHOD_ORDER
    x = np.arange(len(methods))

    expert_vals  = np.array([data[m][0] for m in methods])
    attempt_comp = np.array([alpha * data[m][1] for m in methods])
    total_cost   = expert_vals + attempt_comp

    # Sort by total cost ascending
    order = np.argsort(total_cost)
    methods_s = [methods[i] for i in order]
    expert_s  = expert_vals[order]
    attempt_s = attempt_comp[order]
    total_s   = total_cost[order]

    gengate_pos = methods_s.index('GenGate')

    ec = [C_EXPERT_GENGATE  if m == 'GenGate' else C_EXPERT_BASE  for m in methods_s]
    ac = [C_ATTEMPT_GENGATE if m == 'GenGate' else C_ATTEMPT_BASE for m in methods_s]

    bars1 = ax.bar(x, expert_s,  color=ec, width=0.6, label='ExpertRate', zorder=3)
    bars2 = ax.bar(x, attempt_s, bottom=expert_s, color=ac, width=0.6,
                   label=r'$\alpha\cdot$AttemptRate', zorder=3)

    # Total cost label on top of each bar
    for i, (tot, m) in enumerate(zip(total_s, methods_s)):
        is_gengate = (m == 'GenGate')
        ax.text(i, tot + 0.8, f'{tot:.0f}',
                ha='center', va='bottom', fontsize=6,
                color=C_EXPERT_GENGATE if is_gengate else '#666666',
                fontweight='bold' if is_gengate else 'normal')

    # Highlight GenGate bar with border
    for bar in [bars1[gengate_pos], bars2[gengate_pos]]:
        bar.set_edgecolor(C_EXPERT_GENGATE)
        bar.set_linewidth(1.5)

    ax.set_xticks(x)
    ax.set_xticklabels([SHORT_LABELS[m] for m in methods_s], fontsize=7.5,
                       rotation=35, ha='right', rotation_mode='anchor')
    ax.set_xlim(-0.6, len(methods) - 0.4)
    ax.set_ylim(0, max(total_s) * 1.18)
    ax.set_title(setting.split('-')[1], fontsize=10, fontweight='bold', pad=4)
    ax.tick_params(axis='y', labelsize=8)
    ax.grid(axis='y', alpha=0.2, lw=0.4, zorder=0)
    ax.set_axisbelow(True)

    # Shade GenGate column
    ax.axvspan(gengate_pos - 0.4, gengate_pos + 0.4, alpha=0.07, color=C_EXPERT_GENGATE, zorder=1)


legend_handles = [
    mpatches.Patch(color=C_EXPERT_BASE,  label='ExpertRate'),
    mpatches.Patch(color=C_ATTEMPT_BASE, label=r'$\alpha\cdot$AttemptRate ($\alpha=0.1$)'),
]

for model, settings, fname in [
    ('Qwen',  ['Qwen-Code',  'Qwen-Math',  'Qwen-MMLU'],  'cost_decomposition_qwen'),
    ('LLaMA', ['LLaMA-Code', 'LLaMA-Math', 'LLaMA-MMLU'], 'cost_decomposition_llama'),
]:
    fig, axes = plt.subplots(1, 3, figsize=(11, 4.2))
    for j, (ax, setting) in enumerate(zip(axes, settings)):
        draw_panel(ax, setting)
        if j == 0:
            ax.set_ylabel(r'Cost$_{0.1}$', fontsize=9)
    fig.legend(
        handles=legend_handles,
        loc='lower center',
        bbox_to_anchor=(0.27, -0.015, 0.46, 0.04),
        bbox_transform=fig.transFigure,
        ncol=2,
        mode='expand',
        fontsize=8.5,
        frameon=False,
        handlelength=1.4,
        handletextpad=0.45,
        borderaxespad=0.0,
    )
    fig.suptitle(
        f'Joint deployment cost at Route@95% ({model} local model)',
        fontsize=10, fontweight='bold', y=1.01)
    plt.tight_layout()
    fig.savefig(out_dir / f'{fname}.pdf', bbox_inches='tight')
    fig.savefig(out_dir / f'{fname}.png', bbox_inches='tight')
    print(f"Saved {fname}.{{pdf,png}}")
    plt.close(fig)
