"""movielens_1_plot -- render the real-graph hero figure.

Reads ``experiments/outputs/movielens_1_results.npz`` (produced by
``movielens_1.py``) and writes ``experiments/outputs/movielens_1.png``.

Single panel: median stopping time vs Laplacian weight rho on a
log-log axis, with 25-75 IQR shading.  TS-Explore and GRUB sweep rho;
Basic TS is broadcast as a horizontal reference.  A vertical dotted
line marks rho^*(eps) predicted by the analysis.
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
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=str,
                        default=os.path.join(OUT, 'movielens_1_results.npz'))
    parser.add_argument('--out', type=str,
                        default=os.path.join(OUT, 'movielens_1.png'))
    args = parser.parse_args()

    if not os.path.exists(args.results):
        print(f"Error: {args.results} not found. "
              f"Run experiments/movielens_1.py first.", file=sys.stderr)
        sys.exit(1)
    z = np.load(args.results, allow_pickle=False)
    rhos = z['rhos']

    fig, ax = plt.subplots(figsize=(8, 5))
    for name, color, marker, ls in ALGOS:
        stop = z[f'{name}_stop']
        if np.all(np.isnan(stop)):
            continue
        med = np.nanmedian(stop, axis=1)
        lo = np.nanpercentile(stop, 25, axis=1)
        hi = np.nanpercentile(stop, 75, axis=1)
        ax.plot(rhos, med, color=color, marker=marker, linestyle=ls,
                label=name, linewidth=2.0, markersize=8)
        ax.fill_between(rhos, lo, hi, color=color, alpha=0.18)

    # rho^* (epsilon) vertical line, using observed TS-Explore median as a T-estimate
    if 'eps' in z.files:
        eps = float(z['eps'])
        K = int(z['K'])
        delta = float(z['delta'])
        med_ts = np.nanmedian(z['TS-Explore_stop'], axis=1)
        T_est = float(np.nanmedian(med_ts)) if med_ts.size else 1e5
        sigma0 = 2.0 * 1.0 * np.sqrt(14.0)
        L1 = np.log(12 * K**2 * max(T_est, 1)**2 / delta)
        if eps > 0:
            rho_star = sigma0 * np.sqrt(L1) / eps
            ax.axvline(rho_star, color='gray', linestyle=':', alpha=0.7,
                       label=fr'$\rho^*(\varepsilon)\approx{rho_star:.1f}$')

    ax.set_xlabel(r'Laplacian weight $\rho$')
    ax.set_ylabel('Stopping time (log scale)')
    ax.set_xscale('log')
    ax.set_yscale('log')
    K = int(z['K']) if 'K' in z.files else None
    eps = float(z['eps']) if 'eps' in z.files else None
    title = 'MovieLens-100K: top-rated BAI on item-item similarity graph'
    if K is not None and eps is not None:
        title += f'\n(K={K}, $\\varepsilon={eps:.2f}$)'
    ax.set_title(title)
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(fontsize=9, loc='best')
    fig.tight_layout()
    fig.savefig(args.out, dpi=150, bbox_inches='tight')
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
