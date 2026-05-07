from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.utils import plotting  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs')


def panel_left(ax, z):
    """Synthetic clustered_chain K-sweep."""
    Ks = z['Ks']
    for name in ['TS-Explore', 'Basic TS', 'KL-LUCB', 'GRUB']:
        key = f'{name}_stop'
        if key not in z.files:
            continue
        st = plotting.style_for(name)
        plotting.plot_with_iqr(ax, Ks, z[key], label=name, **st)
    ax.set_xlabel(r'arms $K$')
    ax.set_ylabel('stopping time')
    ax.set_yscale('log')
    ax.set_xticks(Ks)
    plotting.grid_only_major(ax)


def panel_right(ax, z):
    """MovieLens rho-sweep. TS-Explore + three baselines."""
    rhos = z['rhos']
    series = ['TS-Explore', 'Basic TS', 'KL-LUCB', 'GRUB']
    for name in series:
        key = f'{name}_stop'
        if key not in z.files:
            continue
        st = plotting.style_for(name)
        plotting.plot_with_iqr(ax, rhos, z[key], label=name, **st)

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
        # Annotate inside the data region (mid-y) so it never collides with
        # a top-of-axes legend.
        ax.text(rho_star * 1.12, 0.45, r'$\rho^{*}(\varepsilon)$',
                transform=ax.get_xaxis_transform(),
                color='gray', fontsize=8, va='center', ha='left')

    ax.set_xlabel(r'Laplacian weight $\rho$')
    ax.set_ylabel('stopping time')
    ax.set_xscale('log')
    ax.set_yscale('log')
    plotting.grid_only_major(ax)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--main2', type=str,
                        default=os.path.join(OUT, 'main_2_results.npz'))
    parser.add_argument('--ml', type=str,
                        default=os.path.join(OUT, 'movielens_1_results.npz'))
    parser.add_argument('--out', type=str,
                        default=os.path.join(OUT, 'fig1.png'))
    args = parser.parse_args()

    if not os.path.exists(args.main2):
        print(f"Error: {args.main2} not found.", file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(args.ml):
        print(f"Error: {args.ml} not found.", file=sys.stderr)
        sys.exit(1)

    plotting.apply_paper_style()
    z2 = np.load(args.main2, allow_pickle=False)
    zm = np.load(args.ml, allow_pickle=False)

    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.6),
                             constrained_layout=True)
    panel_left(axes[0], z2)
    panel_right(axes[1], zm)

    plotting.legend_above_figure(fig, axes, y=1.0)

    for p in plotting.save_figure(fig, args.out):
        print(f"Saved {p}")


if __name__ == "__main__":
    main()
