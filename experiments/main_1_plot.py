"""main_1_plot -- agreement-vs-elimination figure (appendix).

Reads ``experiments/outputs/main_1_results.npz`` (produced by
``main_1.py``) and writes ``experiments/outputs/main_1.png``.

Two panels:
  (a) median stopping time vs K, with 25-75 IQR shading.
  (b) single-seed candidate-set count vs time at one K.
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

ALGOS = ['TS-Explore', 'Basic TS', 'GRUB']


def _panel_label(ax, letter):
    ax.text(0.02, 0.96, letter, transform=ax.transAxes,
            fontsize=10, fontweight='bold', va='top', ha='left')


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

    plotting.apply_paper_style()
    z = np.load(args.results, allow_pickle=False)
    Ks = z['Ks']

    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.7),
                             constrained_layout=True)

    # Panel (a): stopping time vs K
    ax = axes[0]
    for name in ALGOS:
        st = plotting.style_for(name)
        plotting.plot_with_iqr(ax, Ks, z[f'{name}_stop'], label=name, **st)
    ax.set_xlabel(r'arms $K$')
    ax.set_ylabel('stopping time')
    ax.set_yscale('log')
    ax.set_xticks(Ks)
    plotting.grid_only_major(ax)
    _panel_label(ax, '(a)')

    # Panel (b): single-seed candidate-set vs time
    ax = axes[1]
    for name in ALGOS:
        st = plotting.style_for(name)
        ts = z[f'{name}_curve_t']
        ns = z[f'{name}_curve_n']
        if ts.size == 0:
            continue
        ax.step(ts, ns, where='post', color=st['color'],
                linestyle='-', label=name, linewidth=1.5)
    ax.set_xlabel(r'time step $t$')
    ax.set_ylabel('remaining candidate arms')
    ax.set_xscale('log')
    plotting.grid_only_major(ax)
    _panel_label(ax, '(b)')

    plotting.legend_above_figure(fig, axes, y=1.0)

    for p in plotting.save_figure(fig, args.out):
        print(f"Saved {p}")


if __name__ == "__main__":
    main()
