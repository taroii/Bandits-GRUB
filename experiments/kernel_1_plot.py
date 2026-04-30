"""kernel_1_plot — render Figure 4 from kernel_1_results.npz.

Reads ``experiments/outputs/kernel_1_results.npz`` (produced by
``kernel_1.py``) and writes ``experiments/outputs/kernel_1.png``.

Single panel: median stopping time vs Laplacian weight rho on a log-log
axis, with 25-75 IQR shading per algorithm.  Vertical dotted lines mark
the predicted optimal rho^*(eps) for each kernel using the median
smoothness over the sweep's instances.
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

# Keep this list in sync with ``kernel_1.py``.
ALGOS = [
    ('TS-K_G',   r'TS-Explore (normalized $K_G$)',     '#1f77b4', 's', '-'),
    ('TS-L_G',   r'TS-Explore (combinatorial $L_G$)',  '#d62728', 'o', '--'),
    ('Basic TS', r'Basic TS (no graph)',                '#2ca02c', '^', ':'),
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

    z = np.load(args.results, allow_pickle=False)
    rhos = z['rhos']

    fig, ax = plt.subplots(figsize=(8, 5))
    for name, label, color, marker, ls in ALGOS:
        plotting.plot_with_ci(ax, rhos, z[f'{name}_stop'], label=label,
                              color=color, marker=marker, ls=ls)

    # Predicted optimal rho^*(eps) = sigma_0 sqrt(L_1) / eps for each kernel.
    # Use the median smoothness across the sweep's instances and the
    # observed median stopping time as a T-estimate for L_1.
    sigma0 = 2.0 * 1.0 * np.sqrt(14.0)
    K = int(z['n'])
    delta = float(z['delta'])
    T_est = float(np.median(z['TS-K_G_stop']))
    L1 = np.log(12 * K**2 * max(T_est, 1)**2 / delta)
    eps_L_med = float(np.nanmedian(z['eps_L']))
    eps_K_med = float(np.nanmedian(z['eps_K']))
    if eps_L_med > 0:
        rho_L_star = sigma0 * np.sqrt(L1) / eps_L_med
        ax.axvline(rho_L_star, color='#d62728', ls=':', alpha=0.5,
                   label=fr'$\rho^*_{{L_G}}\approx{rho_L_star:.1f}$')
    if eps_K_med > 0:
        rho_K_star = sigma0 * np.sqrt(L1) / eps_K_med
        ax.axvline(rho_K_star, color='#1f77b4', ls=':', alpha=0.5,
                   label=fr'$\rho^*_{{K_G}}\approx{rho_K_star:.1f}$')

    ax.set_xlabel(r'Laplacian weight $\rho$')
    ax.set_ylabel('stopping time (log scale)')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_title(f'Kernel comparison on BA hub-challenger '
                 f'(n={int(z["n"])}, m={int(z["m"])}, '
                 f'$\\Delta_{{\\mathrm{{chal}}}}={float(z["gap_chal"])}$)')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which='both')

    fig.tight_layout()
    fig.savefig(args.out, dpi=150, bbox_inches='tight')
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
