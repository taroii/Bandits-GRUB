from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.utils import instances, plotting  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs')

ALGOS = [
    ('TS_tuned', r'TS ($\rho^{*}$ tuned)'),
    ('TS_rho1',  r'TS ($\rho{=}1$)'),
    ('Basic',    r'Basic TS'),
]


def _panel_label(ax, letter):
    ax.text(0.015, 0.96, letter, transform=ax.transAxes,
            fontsize=10, fontweight='bold', va='top', ha='left')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=str,
                        default=os.path.join(OUT, 'mis_1_results.npz'))
    parser.add_argument('--out', type=str,
                        default=os.path.join(OUT, 'mis_1.png'))
    parser.add_argument('--instance-seed', type=int, default=0)
    parser.add_argument('--delta', type=float, default=1e-3)
    args = parser.parse_args()

    if not os.path.exists(args.results):
        print(f"Error: {args.results} not found. "
              f"Run experiments/mis_1.py first.", file=sys.stderr)
        sys.exit(1)

    plotting.apply_paper_style()
    z = np.load(args.results, allow_pickle=False)
    eps = z['eps']
    log_eps = np.log10(eps)

    mu, A, D = instances.sbm_phase_transition(seed=args.instance_seed)
    K = len(mu)
    a_star = int(np.argmax(mu))
    gap2_nonstar = [(mu[a_star] - mu[i]) ** 2
                    for i in range(K) if i != a_star]
    asym = (1.0 / min(gap2_nonstar)) * np.log(1.0 / args.delta)
    sum_limit = sum(1.0 / g for g in gap2_nonstar)
    max_limit = 1.0 / min(gap2_nonstar)

    fig, axes = plt.subplots(3, 1, figsize=(5.2, 6.2), sharex=True,
                             gridspec_kw={'height_ratios': [2.0, 1.2, 1.0]},
                             constrained_layout=True)

    # Panel (a): stopping time
    ax = axes[0]
    for name, label in ALGOS:
        st = plotting.style_for(name)
        plotting.plot_with_iqr(ax, log_eps, z[f'{name}_stop'],
                               label=label, **st)
    eps_critical = z['eps_critical']
    for v in eps_critical:
        if np.isfinite(v) and v > 0:
            ax.axvline(np.log10(v), color='gray', alpha=0.25,
                       linestyle=':', linewidth=0.8, zorder=0)
    ax.axhline(asym, color='black', linestyle='--', alpha=0.6,
               linewidth=1.0,
               label=r'$\max_i\Delta_i^{-2}\log(1/\delta)$')
    ax.set_ylabel('stopping time')
    ax.set_yscale('log')
    plotting.grid_only_major(ax)
    plotting.legend_above(ax, ncol=4)
    _panel_label(ax, '(a)')

    # Panel (b): analytical H_eps
    ax = axes[1]
    ax.step(log_eps, z['H_eps'], where='mid', color='#0072B2',
            linewidth=1.6, label=r'$H_{\varepsilon}$')
    ax.axhline(sum_limit, color='gray', linestyle='--', alpha=0.7,
               linewidth=1.0,
               label=fr'$\sum\Delta^{{-2}}={sum_limit:.0f}$')
    ax.axhline(max_limit, color='black', linestyle='--', alpha=0.7,
               linewidth=1.0,
               label=fr'$\max\Delta^{{-2}}={max_limit:.0f}$')
    ax.set_ylabel(r'$H_{\varepsilon}$')
    plotting.grid_only_major(ax)
    plotting.legend_above(ax, ncol=3)
    _panel_label(ax, '(b)')

    # Panel (c): |competitive set|
    ax = axes[2]
    ax.step(log_eps, z['comp_size'], where='mid', color='#009E73',
            linewidth=1.6)
    ax.set_xlabel(r'$\log_{10}\varepsilon$')
    ax.set_ylabel(r'$|\mathcal{H}_{\varepsilon}|$')
    plotting.grid_only_major(ax)
    _panel_label(ax, '(c)')

    for p in plotting.save_figure(fig, args.out):
        print(f"Saved {p}")


if __name__ == "__main__":
    main()
