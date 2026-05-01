"""main_2_plot -- render Figure for the graph-helpful K-sweep.

Reads ``experiments/outputs/main_2_results.npz`` (produced by
``main_2.py``) and writes ``experiments/outputs/main_2.png``.

Single panel: median stopping time vs K on a log y-axis, with 25-75
IQR shading per algorithm.  The visual signal is whether Basic TS
diverges upward away from TS-Explore as K grows --- the empirical
benefit of the Laplacian-regularized estimator.
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
                        default=os.path.join(OUT, 'main_2_results.npz'))
    parser.add_argument('--out', type=str,
                        default=os.path.join(OUT, 'main_2.png'))
    args = parser.parse_args()

    if not os.path.exists(args.results):
        print(f"Error: {args.results} not found. "
              f"Run experiments/main_2.py first.", file=sys.stderr)
        sys.exit(1)
    z = np.load(args.results, allow_pickle=False)
    Ks = z['Ks']

    fig, ax = plt.subplots(figsize=(8, 5))
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
    C = int(z['C']) if 'C' in z.files else None
    gap_step = float(z['gap_step']) if 'gap_step' in z.files else None
    title = 'Graph-helpful sample complexity on clustered_chain'
    if C is not None and gap_step is not None:
        title += f' (C={C}, $\\Delta_{{\\mathrm{{step}}}}={gap_step}$)'
    ax.set_title(title)
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(loc='best')

    fig.tight_layout()
    fig.savefig(args.out, dpi=150, bbox_inches='tight')
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
