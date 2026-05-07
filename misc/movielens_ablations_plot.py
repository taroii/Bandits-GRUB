from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.utils import plotting  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--results', type=str,
        default=os.path.join(OUT, 'movielens_ablations_results.npz'))
    parser.add_argument(
        '--out', type=str,
        default=os.path.join(OUT, 'movielens_ablations.png'))
    args = parser.parse_args()

    if not os.path.exists(args.results):
        print(f"Error: {args.results} not found.", file=sys.stderr)
        sys.exit(1)

    plotting.apply_paper_style()
    z = np.load(args.results, allow_pickle=False)
    labels = list(z['labels'].tolist())
    algos = list(z['algo_names'].tolist())

    n_cfg = len(labels)
    n_algo = len(algos)
    x = np.arange(n_cfg)
    width = 0.78 / max(n_algo, 1)

    fig, ax = plt.subplots(figsize=(max(4.8, 0.95 * n_cfg), 2.9),
                           constrained_layout=True)
    for ai, a in enumerate(algos):
        stop = z[f'{a}_stop']
        med = np.nanmedian(stop, axis=1)
        lo = np.nanpercentile(stop, 25, axis=1)
        hi = np.nanpercentile(stop, 75, axis=1)
        err = np.array([med - lo, hi - med])
        st = plotting.style_for(a)
        ax.bar(x + (ai - (n_algo - 1) / 2) * width, med, width,
               yerr=err, label=a, color=st['color'],
               capsize=2.5, alpha=0.9, linewidth=0,
               error_kw=dict(linewidth=0.8, ecolor='black'))

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha='right')
    ax.set_yscale('log')
    ax.set_ylabel('stopping time')
    plotting.grid_only_major(ax)
    ax.grid(axis='x', visible=False)
    plotting.legend_above(ax, ncol=n_algo)

    for p in plotting.save_figure(fig, args.out):
        print(f"Saved {p}")
    print()

    # Markdown table for the appendix.
    print("| config | " + " | ".join(algos) + " |")
    print("|" + "---|" * (n_algo + 1))
    for ci, label in enumerate(labels):
        row = [label]
        for a in algos:
            stop = z[f'{a}_stop']
            row.append(f"{np.nanmedian(stop[ci, :]):,.0f}")
        print("| " + " | ".join(row) + " |")


if __name__ == "__main__":
    main()
