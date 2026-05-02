"""chembl_1_plot -- render Figure for the real-graph ChEMBL experiment.

Reads ``experiments/outputs/chembl_1_results.npz`` and writes
``experiments/outputs/chembl_1.png``.

Two-panel figure:
  A) Per-algorithm stopping-time distribution (boxplot or strip plot
     across seeds), with the medians annotated.
  B) The empirical pIC50 distribution of the K molecules in the
     instance (as a histogram), with the best arm marked, so the
     reader can see the gap structure of the underlying problem.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs')

ALGOS = ['TS-Explore', 'Basic TS', 'GRUB']
COLOURS = {
    'TS-Explore': '#d62728',
    'Basic TS':   '#2ca02c',
    'GRUB':       '#1f77b4',
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=str,
                        default=os.path.join(OUT, 'chembl_1_results.npz'))
    parser.add_argument('--out', type=str,
                        default=os.path.join(OUT, 'chembl_1.png'))
    args = parser.parse_args()

    if not os.path.exists(args.results):
        print(f"Error: {args.results} not found. "
              f"Run experiments/chembl_1.py first.", file=sys.stderr)
        sys.exit(1)
    z = np.load(args.results, allow_pickle=False)
    target = str(z['target'])
    K = int(z['K'])
    rho = float(z['rho'])
    mu = z['mu']
    best_arm = int(z['best_arm'])

    # -- Panel A: per-algorithm stopping time across seeds.
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    data = []
    labels = []
    medians = []
    for name in ALGOS:
        ts = z[f'{name}_stop']
        ts = ts[~np.isnan(ts)]
        if ts.size == 0:
            continue
        data.append(ts)
        labels.append(name)
        medians.append(float(np.median(ts)))
    bp = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.55,
                    showmeans=False, medianprops=dict(color='black',
                                                      linewidth=2.0))
    for patch, name in zip(bp['boxes'], labels):
        patch.set_facecolor(COLOURS[name])
        patch.set_alpha(0.55)
    for i, m in enumerate(medians):
        ax.text(i + 1, m, f' {m:.0f}', va='center', ha='left',
                fontsize=9, color='black')
    ax.set_yscale('log')
    ax.set_ylabel('Stopping time (log scale)')
    ax.set_title(f'A. {target} stopping time '
                 f'(K = {K}, $\\rho$ = {rho:.0f})')
    ax.grid(True, which='both', alpha=0.3)

    # -- Panel B: pIC50 histogram with best arm marked.
    ax = axes[1]
    ax.hist(mu, bins=30, color='#888888', alpha=0.75, edgecolor='white')
    ax.axvline(mu[best_arm], color='red', linestyle='--', linewidth=2.0,
               label=f'best arm $\\mu = {mu[best_arm]:.3f}$')
    ax.set_xlabel('Normalized pIC50')
    ax.set_ylabel('# molecules')
    ax.set_title(f'B. Reward distribution across the {K} arms')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"Real-graph experiment on ChEMBL target {target}: "
                 f"TS-Explore vs Basic TS vs GRUB",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150, bbox_inches='tight')
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
