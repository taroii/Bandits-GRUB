"""Task 2 -- plot the smoothness x feedback result.

Left panel : stopping time vs rho at a fixed density, showing that the TS
             agreement rule is ~rho-invariant while UCB-LCB degrades.
Right panel: the TS/UCB ratio vs rho for each density; the horizontal line at
             1 is the break-even, and the 'emp' value is drawn as an open
             marker at the left edge for reference (it has no rho).

  python experiments/fb_smooth_plot.py [--instance er_smooth --n 20]
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.utils import plotting  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--instance', default='er_smooth')
    ap.add_argument('--n', type=int, default=20)
    ap.add_argument('--panel-p', type=float, default=0.2)
    args = ap.parse_args()

    f = os.path.join(OUT, f'fb_smooth_{args.instance}_n{args.n}_results.npz')
    if not os.path.exists(f):
        print(f"missing {f}")
        return 1
    z = np.load(f, allow_pickle=False)
    ps, rhos = z['ps'], z['rhos']
    combos = [str(c) for c in z['combos']]
    reg, emp = z['reg'], z['emp']
    reg_done = z['reg_done'].astype(bool)
    ci_ts, ci_ucb = combos.index('TS+cover'), combos.index('UCB+cover')

    plotting.apply_paper_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.9))

    # --- left: absolute stopping times at one density ------------------
    pi = int(np.argmin(np.abs(ps - args.panel_p)))
    m = reg_done[pi]
    ax = axes[0]
    for ci, lab, col, mk in [
            (ci_ts, 'TS+cover (agreement stop)', '#E69F00', 's'),
            (ci_ucb, 'UCB+cover (UCB-LCB stop)', '#0072B2', 'D')]:
        med = np.nanmedian(reg[pi, :, ci, :], axis=1)
        lo = np.nanpercentile(reg[pi, :, ci, :], 25, axis=1)
        hi = np.nanpercentile(reg[pi, :, ci, :], 75, axis=1)
        ax.plot(rhos[m], med[m], color=col, marker=mk, label=lab, zorder=3)
        ax.fill_between(rhos[m], lo[m], hi[m], color=col, alpha=0.16,
                        linewidth=0, zorder=2)
        # empirical-estimator reference (rho-independent)
        e = float(np.nanmedian(emp[pi, ci]))
        ax.axhline(e, color=col, linestyle=':', linewidth=1.1, alpha=0.8,
                   zorder=1)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'Laplacian weight $\rho$')
    ax.set_ylabel('stopping time')
    ax.text(0.03, 0.05, f'ER $p={ps[pi]:g}$, $n={args.n}$\n'
            'dotted = unbiased (empirical) estimator',
            transform=ax.transAxes, fontsize=7, va='bottom', alpha=0.8)
    plotting.grid_only_major(ax)

    # --- right: TS/UCB ratio vs rho, one curve per density ------------
    ax = axes[1]
    cmap = plt.get_cmap('viridis')
    for j, p in enumerate(ps):
        mm = reg_done[j]
        if not mm.any():
            continue
        r = (np.nanmedian(reg[j, :, ci_ts, :], axis=1)
             / np.nanmedian(reg[j, :, ci_ucb, :], axis=1))
        col = cmap(0.12 + 0.76 * j / max(len(ps) - 1, 1))
        ax.plot(rhos[mm], r[mm], color=col, marker='o', markersize=4,
                label=f'$p={p:g}$', zorder=3)
        e = (float(np.nanmedian(emp[j, ci_ts]))
             / float(np.nanmedian(emp[j, ci_ucb])))
        if not np.isfinite(e):
            continue
        ax.plot([rhos[mm][0] * 0.45], [e], color=col, marker='o',
                markersize=5, markerfacecolor='white', markeredgewidth=1.2,
                zorder=4)
    ax.axhline(1.0, color='k', linestyle='--', linewidth=0.9, alpha=0.6)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'Laplacian weight $\rho$')
    ax.set_ylabel(r'TS / UCB stopping-time ratio')
    ax.text(0.03, 0.05, 'hollow = unbiased estimator\nbelow 1 = TS wins',
            transform=ax.transAxes, fontsize=7, va='bottom', alpha=0.8)
    plotting.grid_only_major(ax)

    axes[0].legend(loc='upper left', fontsize=7.5)
    axes[1].legend(loc='upper right', fontsize=7, ncol=1)
    fig.tight_layout()
    for w in plotting.save_figure(fig, os.path.join(OUT, 'fb_smooth.pdf')):
        print("wrote", w)

    # --- text table ---------------------------------------------------
    print("\n# TS+cover / UCB+cover stopping-time ratio (<1 = TS wins)")
    hdr = f"{'p':>6} {'emp':>8}" + "".join(f"{'r=' + f'{r:g}':>10}" for r in rhos)
    print(hdr)
    for j, p in enumerate(ps):
        if not np.isfinite(np.nanmedian(emp[j, ci_ts])):
            continue          # density not run yet
        e = (float(np.nanmedian(emp[j, ci_ts]))
             / float(np.nanmedian(emp[j, ci_ucb])))
        row = f"{p:>6g} {e:>8.3f}"
        for k in range(len(rhos)):
            if reg_done[j, k]:
                r = (float(np.nanmedian(reg[j, k, ci_ts]))
                     / float(np.nanmedian(reg[j, k, ci_ucb])))
                row += f"{r:>10.3f}"
            else:
                row += f"{'--':>10}"
        print(row)
    return 0


if __name__ == '__main__':
    sys.exit(main())
