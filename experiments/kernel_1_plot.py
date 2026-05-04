"""kernel_1_plot -- normalized vs combinatorial Laplacian rho-sweep
(appendix figure).

Reads ``experiments/outputs/kernel_1_results.npz`` and writes
``experiments/outputs/kernel_1.png``.
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

ALGOS = [
    ('TS-K_G',   r'TS-Explore (normalized $K_G$)'),
    ('TS-L_G',   r'TS-Explore (combinatorial $L_G$)'),
    ('Basic TS', r'Basic TS'),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=str,
                        default=os.path.join(OUT, 'kernel_1_results.npz'))
    parser.add_argument('--out', type=str,
                        default=os.path.join(OUT, 'kernel_1.png'))
    args = parser.parse_args()

    if not os.path.exists(args.results):
        print(f"Error: {args.results} not found. "
              f"Run experiments/kernel_1.py first.", file=sys.stderr)
        sys.exit(1)

    plotting.apply_paper_style()
    z = np.load(args.results, allow_pickle=False)
    rhos = z['rhos']

    fig, ax = plt.subplots(figsize=(5.2, 2.9), constrained_layout=True)
    for name, label in ALGOS:
        st = plotting.style_for(name)
        plotting.plot_with_iqr(ax, rhos, z[f'{name}_stop'],
                               label=label, **st)

    # rho^*(eps) annotations -- in axis text, no legend entry, placed mid-y
    # so they never collide with a top-of-axes legend.
    sigma0 = 2.0 * np.sqrt(14.0)
    K = int(z['n'])
    delta = float(z['delta'])
    T_est = float(np.median(z['TS-K_G_stop']))
    L1 = np.log(12 * K ** 2 * max(T_est, 1) ** 2 / delta)
    eps_L_med = float(np.nanmedian(z['eps_L']))
    eps_K_med = float(np.nanmedian(z['eps_K']))
    if eps_L_med > 0:
        rho_L_star = sigma0 * np.sqrt(L1) / eps_L_med
        col = plotting.style_for('TS-L_G')['color']
        ax.axvline(rho_L_star, color=col, linestyle=':',
                   alpha=0.55, linewidth=1.0, zorder=0)
        ax.text(rho_L_star * 1.10, 0.35, r'$\rho^{*}_{L_G}$',
                transform=ax.get_xaxis_transform(),
                color=col, fontsize=8, va='center', ha='left')
    if eps_K_med > 0:
        rho_K_star = sigma0 * np.sqrt(L1) / eps_K_med
        col = plotting.style_for('TS-K_G')['color']
        ax.axvline(rho_K_star, color=col, linestyle=':',
                   alpha=0.55, linewidth=1.0, zorder=0)
        ax.text(rho_K_star * 1.10, 0.20, r'$\rho^{*}_{K_G}$',
                transform=ax.get_xaxis_transform(),
                color=col, fontsize=8, va='center', ha='left')

    ax.set_xlabel(r'Laplacian weight $\rho$')
    ax.set_ylabel('stopping time')
    ax.set_xscale('log')
    ax.set_yscale('log')
    plotting.grid_only_major(ax)
    plotting.legend_above(ax, ncol=3)

    for p in plotting.save_figure(fig, args.out):
        print(f"Saved {p}")


if __name__ == "__main__":
    main()
