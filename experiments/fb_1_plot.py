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
# TS+cover is TS-Explore-GF (lives in fb_1_results.npz);
# UCB+width is UCB-N (lives in fb_1_results.npz);
# TS+width and UCB+cover live in fb_ablation_results.npz.
ABLATION_ALGOS = ['TS+cover', 'TS+width', 'UCB+cover', 'UCB+width']
ABLATION_LABEL = {
    'TS+cover':  'TS-stop, cover-pair pull',
    'TS+width':  'TS-stop, width pull',
    'UCB+cover': 'UCB-stop, cover-pair pull',
    'UCB+width': 'UCB-stop, width pull',
}


def panel_headline(ax, z, z_abl):
    """Headline panel reads UCB+cover from the ablation file but the
    rest from the main fb_1 file."""
    ps = z['ps']
    for name in HEADLINE_ALGOS:
        if name == 'UCB+cover':
            key = 'UCB+cover_stop'
            if z_abl is None or key not in z_abl.files:
                continue
            data = z_abl[key]
        else:
            key = f'{name}_stop'
            if key not in z.files:
                continue
            data = z[key]
        st = plotting.style_for(name)
        plotting.plot_with_iqr(ax, ps, data, label=name, **st)
    ax.set_xlabel(r'edge probability $p$')
    ax.set_ylabel('stopping time')
    ax.set_yscale('log')
    plotting.grid_only_major(ax)


def _ablation_data(name, z, z_abl):
    """Resolve the npz array for an ablation corner.

    ``TS+cover`` is identical to ``TS-Explore-GF`` (lives in fb_1) and
    ``UCB+width`` is identical to ``UCB-N`` (also fb_1). The other two
    corners are only in the ablation file.
    """
    if name == 'TS+cover':
        return z['TS-Explore-GF_stop']
    if name == 'UCB+width':
        return z['UCB-N_stop']
    if z_abl is None:
        return None
    key = f'{name}_stop'
    return z_abl[key] if key in z_abl.files else None


def panel_ablation(ax, z, z_abl):
    ps = z['ps']
    for name in ABLATION_ALGOS:
        data = _ablation_data(name, z, z_abl)
        if data is None:
            continue
        st = plotting.style_for(name)
        plotting.plot_with_iqr(ax, ps, data, label=ABLATION_LABEL[name], **st)
    ax.set_xlabel(r'edge probability $p$')
    ax.set_ylabel('stopping time')
    ax.set_yscale('log')
    plotting.grid_only_major(ax)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=str,
                        default=os.path.join(OUT, 'fb_1_results.npz'),
                        help="fb_1 headline results (from fb_1.py)")
    parser.add_argument('--ablation', type=str,
                        default=os.path.join(OUT, 'fb_ablation_results.npz'),
                        help="2x2 ablation results (from fb_ablation.py)")
    parser.add_argument('--out', type=str,
                        default=os.path.join(OUT, 'fb_1.png'))
    args = parser.parse_args()

    if not os.path.exists(args.results):
        print(f"Error: {args.results} not found. "
              f"Run experiments/fb_1.py first.", file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(args.ablation):
        print(f"Warning: {args.ablation} not found. "
              f"Right panel will be empty for ablation-only series. "
              f"Run experiments/fb_ablation.py to populate it.",
              file=sys.stderr)

    plotting.apply_paper_style()
    z = np.load(args.results, allow_pickle=False)
    z_abl = (np.load(args.ablation, allow_pickle=False)
             if os.path.exists(args.ablation) else None)

    fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.2),
                             constrained_layout=True)
    panel_headline(axes[0], z, z_abl)
    panel_ablation(axes[1], z, z_abl)

    # Per-panel legends placed below to avoid overlap.
    axes[0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.22),
                   fontsize=7, ncol=2, frameon=False)
    axes[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.22),
                   fontsize=7, ncol=2, frameon=False)

    for p in plotting.save_figure(fig, args.out):
        print(f"Saved {p}")


if __name__ == "__main__":
    main()
