"""movielens_1_plot -- single-panel render of the MovieLens rho-sweep.

Reads ``experiments/outputs/movielens_1_results.npz`` and writes
``experiments/outputs/movielens_1.png``. The combined paper Figure 1 is
produced by ``fig1_plot.py``; this script is kept for standalone /
appendix use and follows the same paper style guide.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.utils import plotting  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs')

ALGOS = ['TS-Explore', 'Basic TS', 'KL-LUCB', 'GRUB']


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

    plotting.apply_paper_style()
    z = np.load(args.results, allow_pickle=False)
    rhos = z['rhos']

    fig, ax = plt.subplots(figsize=(3.6, 2.6), constrained_layout=True)
    for name in ALGOS:
        key = f'{name}_stop'
        if key not in z.files:
            continue
        if np.all(np.isnan(z[key])):
            continue
        st = plotting.style_for(name)
        plotting.plot_with_iqr(ax, rhos, z[key], label=name, **st)

    if 'eps' in z.files:
        eps = float(z['eps'])
        K = int(z['K'])
        delta = float(z['delta'])
        med_ts = np.nanmedian(z['TS-Explore_stop'], axis=1)
        T_est = float(np.nanmedian(med_ts)) if med_ts.size else 1e5
        sigma0 = 2.0 * np.sqrt(14.0)
        L1 = np.log(12 * K ** 2 * max(T_est, 1) ** 2 / delta)
        rho_star = sigma0 * np.sqrt(L1) / eps if eps > 0 else None
        if rho_star is not None and rho_star > 0:
            ax.axvline(rho_star, color='gray', linestyle=':',
                       linewidth=1.0, alpha=0.7, zorder=0)
            ax.text(rho_star * 1.12, 0.45, r'$\rho^{*}(\varepsilon)$',
                    transform=ax.get_xaxis_transform(),
                    color='gray', fontsize=8, va='center', ha='left')

    ax.set_xlabel(r'Laplacian weight $\rho$')
    ax.set_ylabel('stopping time')
    ax.set_xscale('log')
    ax.set_yscale('log')
    plotting.grid_only_major(ax)
    plotting.legend_above(ax, ncol=4)

    for p in plotting.save_figure(fig, args.out):
        print(f"Saved {p}")


if __name__ == "__main__":
    main()
