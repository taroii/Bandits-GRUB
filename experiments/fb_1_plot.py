"""fb_1_plot -- render the graph-feedback density-sweep figure.

Reads ``experiments/outputs/fb_1_results.npz`` and writes
``experiments/outputs/fb_1.{pdf,png}``.

Two panels:
  Left:  headline comparison -- TS-Explore-GF vs. UCB+cover (the
         de-confounded UCB-LCB baseline using the same cover-pair pull
         rule), KL-LUCB (no-graph BAI), TS-Explore at rho=1, Basic TS.
  Right: 2x2 stop-rule x pull-rule decomposition.

The headline panel uses colors keyed to the algorithm; the ablation
panel uses color for the stopping rule and marker for the pull rule.
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

HEADLINE_ALGOS = ['TS-Explore-GF', 'UCB+cover', 'KL-LUCB', 'TS-Explore', 'Basic TS']
# TS+cover is TS-Explore-GF; UCB+width is UCB-N. Map aliases to the
# canonical npz keys present in fb_1_results.npz.
ABLATION_ALGOS = ['TS+cover', 'TS+width', 'UCB+cover', 'UCB+width']
ABLATION_KEY = {
    'TS+cover':  'TS-Explore-GF_stop',
    'TS+width':  'TS+width_stop',
    'UCB+cover': 'UCB+cover_stop',
    'UCB+width': 'UCB-N_stop',
}
ABLATION_LABEL = {
    'TS+cover':  'TS-stop, cover-pair pull',
    'TS+width':  'TS-stop, width pull',
    'UCB+cover': 'UCB-stop, cover-pair pull',
    'UCB+width': 'UCB-stop, width pull',
}


def panel_headline(ax, z):
    ps = z['ps']
    for name in HEADLINE_ALGOS:
        key = f'{name}_stop'
        if key not in z.files:
            continue
        st = plotting.style_for(name)
        plotting.plot_with_iqr(ax, ps, z[key], label=name, **st)
    ax.set_xlabel(r'edge probability $p$')
    ax.set_ylabel('stopping time')
    ax.set_yscale('log')
    plotting.grid_only_major(ax)


def panel_ablation(ax, z):
    ps = z['ps']
    # Two visual axes: color = stopping rule (orange = TS, blue = UCB),
    # marker = pull rule (square = cover-pair, diamond = width).
    for name in ABLATION_ALGOS:
        key = ABLATION_KEY[name]
        if key not in z.files:
            continue
        st = plotting.style_for(name)
        plotting.plot_with_iqr(ax, ps, z[key], label=ABLATION_LABEL[name], **st)
    ax.set_xlabel(r'edge probability $p$')
    ax.set_ylabel('stopping time')
    ax.set_yscale('log')
    plotting.grid_only_major(ax)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=str,
                        default=os.path.join(OUT, 'fb_1_results.npz'))
    parser.add_argument('--out', type=str,
                        default=os.path.join(OUT, 'fb_1.png'))
    args = parser.parse_args()

    if not os.path.exists(args.results):
        print(f"Error: {args.results} not found. "
              f"Run experiments/fb_1.py first.", file=sys.stderr)
        sys.exit(1)

    plotting.apply_paper_style()
    z = np.load(args.results, allow_pickle=False)

    fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.2),
                             constrained_layout=True)
    panel_headline(axes[0], z)
    panel_ablation(axes[1], z)

    # Per-panel legends placed below to avoid overlap.
    axes[0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.22),
                   fontsize=7, ncol=2, frameon=False)
    axes[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.22),
                   fontsize=7, ncol=2, frameon=False)

    for p in plotting.save_figure(fig, args.out):
        print(f"Saved {p}")


if __name__ == "__main__":
    main()
