"""chembl_2_plot -- multi-panel rho-sweep figure for chembl_2 results.

Reads ``experiments/outputs/chembl_2_results.npz`` (produced by
``chembl_2.py``) and writes ``experiments/outputs/chembl_2.png``.

Layout: one row per ChEMBL target.  Each panel plots median stopping
time vs rho on a log--log axis with 25-75% IQR shading, four curves
(TS-Explore, Basic TS, GRUB, KL-LUCB).  Basic TS and KL-LUCB are
rho-free so their curves are flat (constants broadcast across rho).
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs')

ALGOS = [
    ('TS-Explore', '#d62728', 's', '-'),
    ('GRUB',       '#1f77b4', 'o', '-.'),
    ('Basic TS',   '#2ca02c', '^', '--'),
    ('KL-LUCB',    '#9467bd', 'D', ':'),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=str,
                        default=os.path.join(OUT, 'chembl_2_results.npz'))
    parser.add_argument('--out', type=str,
                        default=os.path.join(OUT, 'chembl_2.png'))
    args = parser.parse_args()

    if not os.path.exists(args.results):
        print(f"Error: {args.results} not found. "
              f"Run experiments/chembl_2.py first.", file=sys.stderr)
        sys.exit(1)
    z = np.load(args.results, allow_pickle=False)
    targets = [str(t) for t in z['targets']]
    rhos = z['rhos']
    n_targets = len(targets)

    fig, axes = plt.subplots(1, n_targets, figsize=(5 * n_targets, 5),
                             sharey=False, squeeze=False)
    axes = axes[0]
    for ti, target in enumerate(targets):
        ax = axes[ti]
        for name, color, marker, ls in ALGOS:
            key = f'{target}__{name}'
            stop = z[f'{key}__stop']
            med = np.nanmedian(stop, axis=1)
            lo = np.nanpercentile(stop, 25, axis=1)
            hi = np.nanpercentile(stop, 75, axis=1)
            ax.plot(rhos, med, color=color, marker=marker, linestyle=ls,
                    label=name, linewidth=2.0, markersize=7)
            ax.fill_between(rhos, lo, hi, color=color, alpha=0.16)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel(r'Laplacian weight $\rho$')
        if ti == 0:
            ax.set_ylabel('Stopping time (log scale)')
        eps_L = float(z['eps_L'][ti])
        H_cls = float(z['H_classical'][ti])
        H_gr = float(z['H_graph'][ti])
        ratio = H_cls / max(H_gr, 1e-9)
        title_suffix = (f"\n$\\varepsilon_L = {eps_L:.1f}$,  "
                        f"$H_{{cls}}/H_G = {ratio:.2f}\\times$")
        ax.set_title(f'{target}{title_suffix}')
        ax.grid(True, which='both', alpha=0.3)
        if ti == 0:
            ax.legend(fontsize=8, loc='best')

    fig.suptitle('Real-graph BAI on ChEMBL: '
                 'rho-sweep across three targets',
                 fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150, bbox_inches='tight')
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
