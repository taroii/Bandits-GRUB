"""fb_1_plot -- render the graph-feedback density-sweep figure.

Reads ``experiments/outputs/fb_1_results.npz`` (produced by ``fb_1.py``)
and writes ``experiments/outputs/fb_1.png``.

Single panel: median stopping time vs ER edge probability ``p`` on a
log y-axis, with 25-75 IQR shading per algorithm. Theoretical hardness
reference lines (``H_GF``, ``H_graph``, ``H_classical``) are *not*
plotted in the paper-figure version; the empirical-to-theory gap is
discussed in prose. Pass ``--with-theory`` for an appendix variant that
adds those lines.
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

# Order matters for legend stacking. Headlines first.
ALGOS = ['TS-Explore-GF', 'UCB-N', 'TS-Explore', 'Basic TS']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=str,
                        default=os.path.join(OUT, 'fb_1_results.npz'))
    parser.add_argument('--out', type=str,
                        default=os.path.join(OUT, 'fb_1.png'))
    parser.add_argument('--with-theory', action='store_true',
                        help='also plot H_classical / H_graph / H_GF '
                             'reference lines (appendix variant)')
    args = parser.parse_args()

    if not os.path.exists(args.results):
        print(f"Error: {args.results} not found. "
              f"Run experiments/fb_1.py first.", file=sys.stderr)
        sys.exit(1)

    plotting.apply_paper_style()
    z = np.load(args.results, allow_pickle=False)
    ps = z['ps']
    delta = float(z['delta'])

    fig, ax = plt.subplots(figsize=(4.6, 2.8), constrained_layout=True)

    for name in ALGOS:
        key = f'{name}_stop'
        if key not in z.files:
            continue
        st = plotting.style_for(name)
        plotting.plot_with_iqr(ax, ps, z[key], label=name, **st)

    if args.with_theory:
        log_delta = np.log(1.0 / delta)
        H_graph_med = np.median(z['H_graph'], axis=1)
        H_GF_med = np.median(z['H_GF'], axis=1)
        H_classical = float(z['H_classical'])
        ax.plot(ps, log_delta * H_GF_med, ':', color='black', alpha=0.6,
                linewidth=1.0,
                label=r'$H_{\mathrm{GF}}\cdot\log(1/\delta)$')
        ax.plot(ps, log_delta * H_graph_med, '--', color='black', alpha=0.6,
                linewidth=1.0,
                label=r'$H_{\mathrm{graph}}\cdot\log(1/\delta)$')
        ax.axhline(log_delta * H_classical, color='black', linestyle='-.',
                   alpha=0.6, linewidth=1.0,
                   label=r'$H_{\mathrm{classical}}\cdot\log(1/\delta)$')

    ax.set_xlabel(r'edge probability $p$')
    ax.set_ylabel('stopping time')
    ax.set_yscale('log')
    plotting.grid_only_major(ax)
    plotting.legend_above(ax)

    for p in plotting.save_figure(fig, args.out):
        print(f"Saved {p}")


if __name__ == "__main__":
    main()
