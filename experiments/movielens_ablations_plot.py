"""movielens_ablations_plot -- render the ablation table/figure.

Reads ``experiments/outputs/movielens_ablations_results.npz`` (produced
by ``movielens_ablations.py``) and writes
``experiments/outputs/movielens_ablations.png``.

Renders a grouped bar plot: x-axis = config label, y-axis = stopping
time (log), one bar per algorithm.  Also prints a Markdown-style table
to stdout for easy copy-paste into the paper appendix.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs')

COLORS = {
    'TS-Explore': '#d62728',
    'GRUB':       '#1f77b4',
    'Basic TS':   '#2ca02c',
    'KL-LUCB':    '#9467bd',
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--results', type=str,
        default=os.path.join(OUT, 'movielens_ablations_results.npz'))
    parser.add_argument(
        '--out', type=str,
        default=os.path.join(OUT, 'movielens_ablations.png'))
    args = parser.parse_args()

    if not os.path.exists(args.results):
        print(f"Error: {args.results} not found.", file=sys.stderr)
        sys.exit(1)
    z = np.load(args.results, allow_pickle=False)
    labels = list(z['labels'].tolist())
    algos = list(z['algo_names'].tolist())
    rho = float(z['rho'])

    # Bar plot.
    n_cfg = len(labels)
    n_algo = len(algos)
    x = np.arange(n_cfg)
    width = 0.8 / max(n_algo, 1)

    fig, ax = plt.subplots(figsize=(max(8, 1.4 * n_cfg), 5))
    for ai, a in enumerate(algos):
        stop = z[f'{a}_stop']
        med = np.nanmedian(stop, axis=1)
        lo = np.nanpercentile(stop, 25, axis=1)
        hi = np.nanpercentile(stop, 75, axis=1)
        err = np.array([med - lo, hi - med])
        ax.bar(x + (ai - (n_algo - 1) / 2) * width, med, width,
               yerr=err, label=a, color=COLORS.get(a, None),
               capsize=3, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha='right')
    ax.set_yscale('log')
    ax.set_ylabel('Stopping time (log scale)')
    ax.set_title(f'MovieLens ablation: stopping time across configs '
                 f'($\\rho={rho:g}$, '
                 f'reward = {str(z["reward_model"])})')
    ax.grid(True, axis='y', which='both', alpha=0.3)
    ax.legend(loc='best')
    fig.tight_layout()
    fig.savefig(args.out, dpi=150, bbox_inches='tight')
    print(f"Saved {args.out}")
    print()

    # Markdown table.
    print("| config | " + " | ".join(algos) + " |")
    print("|" + "---|" * (n_algo + 1))
    for ci, label in enumerate(labels):
        row = [label]
        for a in algos:
            stop = z[f'{a}_stop']
            row.append(f"{np.nanmedian(stop[ci, :]):,.0f}")
        print("| " + " | ".join(row) + " |")


if __name__ == "__main__":
    main()
