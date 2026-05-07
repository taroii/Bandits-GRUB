from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.utils import plotting  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs')


# Color is the stopping rule; marker is the pull rule.
ALGO_STYLE = {
    'TS+cover':  {'color': '#D55E00', 'marker': 's',  # vermillion
                  'label': r'TS-stop, cover-pair pull'},
    'TS+width':  {'color': '#D55E00', 'marker': 'D',
                  'label': r'TS-stop, width pull'},
    'UCB+cover': {'color': '#56B4E9', 'marker': 's',  # sky blue
                  'label': r'UCB-stop, cover-pair pull'},
    'UCB+width': {'color': '#56B4E9', 'marker': 'D',
                  'label': r'UCB-stop, width pull'},
}

# Plot order (legend order).  Keep the headlines first.
PLOT_ORDER = ['TS+cover', 'TS+width', 'UCB+cover', 'UCB+width']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=str,
                        default=os.path.join(OUT, 'fb_ablation_results.npz'))
    parser.add_argument('--out', type=str,
                        default=os.path.join(OUT, 'fb_ablation.png'))
    args = parser.parse_args()

    if not os.path.exists(args.results):
        print(f"Error: {args.results} not found. "
              f"Run experiments/fb_ablation.py first.", file=sys.stderr)
        sys.exit(1)

    plotting.apply_paper_style()
    z = np.load(args.results, allow_pickle=False)
    ps = z['ps']

    fig, ax = plt.subplots(figsize=(4.8, 2.9), constrained_layout=True)

    for name in PLOT_ORDER:
        key = f'{name}_stop'
        if key not in z.files:
            continue
        st = ALGO_STYLE[name]
        plotting.plot_with_iqr(ax, ps, z[key],
                               label=st['label'],
                               color=st['color'], marker=st['marker'])

    ax.set_xlabel(r'edge probability $p$')
    ax.set_ylabel('stopping time')
    ax.set_yscale('log')
    plotting.grid_only_major(ax)
    plotting.legend_above(ax, ncol=2)

    for p in plotting.save_figure(fig, args.out):
        print(f"Saved {p}")


if __name__ == "__main__":
    main()
