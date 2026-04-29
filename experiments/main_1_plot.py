"""main_1_plot — render Figure 1 from main_1_results.npz.

Reads ``experiments/outputs/main_1_results.npz`` (produced by
``main_1.py``) and writes ``experiments/outputs/main_1.png``.

Two panels:
  * A — median stopping time vs K, log y-axis, shaded 25/75 IQR.
  * B — single-seed candidate-set / agreement curve at one K.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs')

ALGOS = ['TS-Explore', 'Basic TS', 'GRUB']
STYLES = {
    'TS-Explore': {'color': '#d62728', 'marker': 's', 'ls': '-'},
    'Basic TS':   {'color': '#2ca02c', 'marker': '^', 'ls': '--'},
    'GRUB':       {'color': '#1f77b4', 'marker': 'o', 'ls': '-.'},
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=str,
                        default=os.path.join(OUT, 'main_1_results.npz'))
    parser.add_argument('--out', type=str,
                        default=os.path.join(OUT, 'main_1.png'))
    args = parser.parse_args()

    if not os.path.exists(args.results):
        print(f"Error: {args.results} not found. "
              f"Run experiments/main_1.py first.", file=sys.stderr)
        sys.exit(1)
    z = np.load(args.results, allow_pickle=False)
    Ks = z['Ks']
    K_b = int(z['panel_b_K'])
    seed_b = int(z['panel_b_seed'])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Panel A: stopping time vs K
    ax = axes[0]
    for name in ALGOS:
        st = STYLES[name]
        stop = z[f'{name}_stop']
        med = np.median(stop, axis=1)
        lo = np.percentile(stop, 25, axis=1)
        hi = np.percentile(stop, 75, axis=1)
        ax.plot(Ks, med, color=st['color'], marker=st['marker'],
                linestyle=st['ls'], label=name, linewidth=2.0, markersize=8)
        ax.fill_between(Ks, lo, hi, color=st['color'], alpha=0.18)
    ax.set_xlabel('Number of arms K')
    ax.set_ylabel('Stopping time (log scale)')
    ax.set_yscale('log')
    ax.set_title('A. Sample complexity vs K (median, 25-75 IQR)')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(loc='best')

    # Panel B: single-seed candidate-set/agreement curves
    ax = axes[1]
    for name in ALGOS:
        st = STYLES[name]
        ts = z[f'{name}_curve_t']
        ns = z[f'{name}_curve_n']
        ax.step(ts, ns, where='post', color=st['color'],
                linestyle=st['ls'], label=name, linewidth=2.0)
    ax.set_xlabel('Time step t')
    ax.set_ylabel('Number of remaining candidate arms')
    ax.set_xscale('log')
    ax.set_title(f'B. Single-seed candidate set vs time '
                 f'(K={K_b}, seed={seed_b})')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(loc='best')

    fig.suptitle("Figure 1: TS-Explore agreement-stopping vs GRUB UCB elimination "
                 "on graph-structured pure exploration",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150, bbox_inches='tight')
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
