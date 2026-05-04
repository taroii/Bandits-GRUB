"""chembl_2_plot -- multi-panel rho-sweep figure for chembl_2 results.

Reads ``experiments/outputs/chembl_2_results.npz`` (produced by
``chembl_2.py``) and writes ``experiments/outputs/chembl_2.png``.

Layout: one row per ChEMBL target.  Each panel plots median stopping
time vs rho on a log--log axis with 25-75% IQR shading, four curves
(TS-Explore, Basic TS, GRUB, KL-LUCB).  Basic TS and KL-LUCB are
rho-free so their curves are flat (constants broadcast across rho).

Cells where the algorithm achieved less than 100% correctness across
seeds are drawn with hollow markers and annotated with the per-cell
correctness rate, so the reader is not misled by fast but wrong
convergences (e.g. TS-Explore on small-gap targets at large rho, where
the regularizer biases the best-arm estimate below the runner-up).
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs')

ALGOS = [
    ('TS-Explore', '#d62728', 's', '-'),
    ('GRUB',       '#1f77b4', 'o', '-.'),
    ('Basic TS',   '#2ca02c', '^', '--'),
    ('KL-LUCB',    '#9467bd', 'D', ':'),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=str,
                        default=os.path.join(OUT, 'chembl_2_results.npz'))
    parser.add_argument('--out', type=str,
                        default=os.path.join(OUT, 'chembl_2.png'))
    args = parser.parse_args()

    if not os.path.exists(args.results):
        print(f"Error: {args.results} not found. "
              f"Run experiments/chembl_2.py first.", file=sys.stderr)
        sys.exit(1)
    z = np.load(args.results, allow_pickle=False)
    targets = [str(t) for t in z['targets']]
    rhos = z['rhos']
    n_targets = len(targets)

    fig, axes = plt.subplots(1, n_targets, figsize=(5 * n_targets, 5),
                             sharey=False, squeeze=False)
    axes = axes[0]
    for ti, target in enumerate(targets):
        ax = axes[ti]
        for name, color, marker, ls in ALGOS:
            key = f'{target}__{name}'
            stop = z[f'{key}__stop']
            corr = z[f'{key}__correct'].astype(float).mean(axis=1)  # per-rho
            med = np.nanmedian(stop, axis=1)
            lo = np.nanpercentile(stop, 25, axis=1)
            hi = np.nanpercentile(stop, 75, axis=1)
            is_ok = corr >= 0.999

            # Line through all rhos so the curve shape is visible.
            ax.plot(rhos, med, color=color, linestyle=ls, linewidth=2.0,
                    label=name, zorder=2)
            # IQR shading is dimmed where correctness < 100%, since the
            # spread there mixes fast-but-wrong with slow-but-right runs.
            if np.all(is_ok):
                ax.fill_between(rhos, lo, hi, color=color, alpha=0.16,
                                zorder=1)
            else:
                # Fill only between adjacent 100%-correct points.
                for i in range(len(rhos) - 1):
                    if is_ok[i] and is_ok[i + 1]:
                        ax.fill_between(rhos[i:i + 2], lo[i:i + 2],
                                        hi[i:i + 2], color=color,
                                        alpha=0.16, zorder=1)
                # Faded shade across the full range as a soft hint.
                ax.fill_between(rhos, lo, hi, color=color, alpha=0.04,
                                zorder=1)

            # Filled markers at 100%-correct cells, hollow elsewhere.
            ax.scatter(rhos[is_ok], med[is_ok], color=color, marker=marker,
                       s=70, zorder=3, edgecolor=color, linewidth=1.5)
            if (~is_ok).any():
                ax.scatter(rhos[~is_ok], med[~is_ok], facecolor='white',
                           edgecolor=color, marker=marker, s=70, zorder=3,
                           linewidth=1.5)
                # Annotate correctness rate near each hollow marker.
                for ri, ok in enumerate(is_ok):
                    if not ok:
                        ax.annotate(f'{corr[ri] * 100:.0f}%',
                                    xy=(rhos[ri], med[ri]),
                                    xytext=(4, 6), textcoords='offset points',
                                    fontsize=7, color=color, zorder=4)

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel(r'Laplacian weight $\rho$')
        if ti == 0:
            ax.set_ylabel('Stopping time (log scale)')
        eps_L = float(z['eps_L'][ti])
        H_cls = float(z['H_classical'][ti])
        H_gr = float(z['H_graph'][ti])
        smallest_gap = float(z['smallest_gap'][ti])
        ratio = H_cls / max(H_gr, 1e-9)
        title_suffix = (f"\n$\\varepsilon_L = {eps_L:.1f}$, "
                        f"$\\Delta_{{\\min}} = {smallest_gap:.2f}$, "
                        f"$H_{{cls}}/H_G = {ratio:.2f}\\times$")
        ax.set_title(f'{target}{title_suffix}')
        ax.grid(True, which='both', alpha=0.3)

    # Single shared legend including the marker convention.
    handles = []
    for name, color, marker, ls in ALGOS:
        handles.append(Line2D([0], [0], color=color, linestyle=ls,
                              marker=marker, markersize=7,
                              markerfacecolor=color, markeredgecolor=color,
                              linewidth=2.0, label=name))
    handles.append(Line2D([0], [0], color='gray', linestyle='None',
                          marker='o', markersize=8, markerfacecolor='white',
                          markeredgecolor='gray', linewidth=1.5,
                          label='< 100% correct'))
    fig.legend(handles=handles, loc='lower center', ncol=len(handles),
               fontsize=9, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle('Real-graph BAI on ChEMBL: '
                 'rho-sweep across three targets',
                 fontsize=12, y=1.02)
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(args.out, dpi=150, bbox_inches='tight')
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
