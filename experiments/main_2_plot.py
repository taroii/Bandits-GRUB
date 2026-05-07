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
                        default=os.path.join(OUT, 'main_2_results.npz'))
    parser.add_argument('--out', type=str,
                        default=os.path.join(OUT, 'main_2.png'))
    args = parser.parse_args()

    if not os.path.exists(args.results):
        print(f"Error: {args.results} not found. "
              f"Run experiments/main_2.py first.", file=sys.stderr)
        sys.exit(1)

    plotting.apply_paper_style()
    z = np.load(args.results, allow_pickle=False)
    Ks = z['Ks']

    fig, ax = plt.subplots(figsize=(3.4, 2.6), constrained_layout=True)
    for name in ALGOS:
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
    plotting.legend_above(ax)

    for p in plotting.save_figure(fig, args.out):
        print(f"Saved {p}")


if __name__ == "__main__":
    main()
