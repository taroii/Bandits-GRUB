"""Task 0 -- plot the GRUB bias conventions side by side.

Two panels, matching the two GRUB cells of Figure 1:
  left  : synthetic chain K-sweep at rho = 100
  right : MovieLens-100K rho-sweep at K = 20

One curve per bias convention.  Cap-hits are drawn as open markers with a
hatched band and annotated, so a cell pinned at the per-run budget is never
read as a stopping time (per the rebuttal ground rules).  The committed
TS-Explore curve is overlaid for scale.

  python experiments/grub_bias_plot.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.utils import plotting  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs')

BIAS_STYLE = {
    'published': {'color': '#D55E00', 'marker': '^',
                  'label': r'GRUB, $\rho\varepsilon$ (Thaker et al.)'},
    'sqrt':      {'color': '#0072B2', 'marker': 's',
                  'label': r'GRUB, $\sqrt{\rho}\,\varepsilon$ (tight)'},
    'legacy':    {'color': '#56B4E9', 'marker': 'o',
                  'label': 'GRUB, as submitted'},
    'none':      {'color': '#555555', 'marker': 'v',
                  'label': 'GRUB, no bias term'},
}
ORDER = ['published', 'legacy', 'sqrt', 'none']


def load(instance, bias):
    p = os.path.join(OUT, f'grub_bias_{instance}_{bias}_results.npz')
    if not os.path.exists(p):
        return None
    z = np.load(p, allow_pickle=False)
    done = z['done'].astype(bool)
    if not done.any():
        return None
    return dict(axis=z['axis'], stop=z['stop'], capped=z['capped'].astype(bool),
                done=done, max_steps=int(z['max_steps']))


def panel(ax, instance, xlabel, annotate_cap=True):
    any_cap = False
    for bias in ORDER:
        d = load(instance, bias)
        if d is None:
            continue
        m = d['done']
        x = d['axis'][m]
        runs = d['stop'][m]
        cap = d['capped'][m]
        st = BIAS_STYLE[bias]
        med = np.nanmedian(runs, axis=1)
        lo = np.nanpercentile(runs, 25, axis=1)
        hi = np.nanpercentile(runs, 75, axis=1)
        fully_capped = cap.all(axis=1)
        any_cap = any_cap or fully_capped.any()
        ax.plot(x, med, color=st['color'], marker=st['marker'],
                label=st['label'], zorder=3)
        ax.fill_between(x, lo, hi, color=st['color'], alpha=0.16,
                        linewidth=0, zorder=2)
        # overplot capped cells as hollow markers
        if fully_capped.any():
            ax.plot(x[fully_capped], med[fully_capped], linestyle='none',
                    marker=st['marker'], markerfacecolor='white',
                    markeredgecolor=st['color'], markeredgewidth=1.4,
                    markersize=8, zorder=4)
    if any_cap:
        d = load(instance, 'published') or load(instance, 'legacy')
        if d:
            ax.axhline(d['max_steps'], color='k', linestyle='--',
                       linewidth=0.9, alpha=0.6, zorder=1)
            if annotate_cap:
                ax.text(0.02, 0.03,
                        'dashed = per-run cap;\nhollow = all seeds capped',
                        transform=ax.transAxes, fontsize=7,
                        color='k', alpha=0.75, va='bottom', ha='left')
    ax.set_xlabel(xlabel)
    ax.set_xscale('log')
    ax.set_yscale('log')
    plotting.grid_only_major(ax)


def main():
    plotting.apply_paper_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.9))

    panel(axes[0], 'chain', r'number of arms $K$')
    axes[0].set_ylabel('stopping time')

    panel(axes[1], 'movielens', r'Laplacian weight $\rho$',
          annotate_cap=False)

    # TS-Explore for scale, from the committed results.
    p = os.path.join(OUT, 'main_2_results.npz')
    if os.path.exists(p):
        z = np.load(p, allow_pickle=False)
        axes[0].plot(z['Ks'], np.nanmedian(z['TS-Explore_stop'], axis=1),
                     color='#009E73', marker='P', linestyle='--',
                     label='TS-Explore (submitted)', zorder=3)
    p = os.path.join(OUT, 'movielens_1_results.npz')
    if os.path.exists(p):
        z = np.load(p, allow_pickle=False)
        axes[1].plot(z['rhos'], np.nanmedian(z['TS-Explore_stop'], axis=1),
                     color='#009E73', marker='P', linestyle='--',
                     label='TS-Explore (submitted)', zorder=3)

    plotting.legend_above_figure(fig, axes, ncol=3, y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    written = plotting.save_figure(fig, os.path.join(OUT, 'grub_bias.pdf'))
    for w in written:
        print("wrote", w)

    # text summary
    print("\n# GRUB medians by bias convention (CAP = all seeds hit the cap)")
    for instance, ax_name in (('chain', 'K'), ('movielens', 'rho')):
        print(f"\n{instance} ({ax_name}-sweep)")
        for bias in ORDER:
            d = load(instance, bias)
            if d is None:
                print(f"  {bias:10s} (no data yet)")
                continue
            m = d['done']
            cells = []
            for j, xv in enumerate(d['axis'][m]):
                med = np.nanmedian(d['stop'][m][j])
                nc = int(d['capped'][m][j].sum())
                tot = d['stop'].shape[1]
                cells.append(f"{ax_name}={xv:g}:"
                             + (f"CAP{nc}/{tot}" if nc == tot
                                else f"{med:,.0f}"
                                + (f"(cap{nc})" if nc else "")))
            print(f"  {bias:10s} " + "  ".join(cells))
    return 0


if __name__ == '__main__':
    sys.exit(main())
