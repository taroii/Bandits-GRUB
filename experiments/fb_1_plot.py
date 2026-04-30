"""fb_1_plot — render Figure 3 from fb_1_results.npz.

Reads ``experiments/outputs/fb_1_results.npz`` (produced by ``fb_1.py``)
and writes ``experiments/outputs/fb_1.png``.

Single-panel figure: median stopping time vs ER edge density ``p`` on a
log y-axis, with 25-75 IQR shading per algorithm and dotted reference
lines for ``H_GF * log(1/delta)``, ``H_graph * log(1/delta)``, and
``H_classical * log(1/delta)`` (using the median ``H`` across seeds).
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

# Keep this list in sync with ``fb_1.py``.
ALGOS = [
    ('TS-Explore-GF', 'TS-Explore-GF',          '#9467bd', 's', '-'),
    ('UCB-N',         'UCB-N (Caron et al.)',   '#1f77b4', 'D', '-'),
    ('TS-Explore',    'TS-Explore (graph)',     '#d62728', 'o', '--'),
    ('Basic TS',      'Basic TS (no graph)',    '#2ca02c', '^', ':'),
]


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

    z = np.load(args.results, allow_pickle=False)
    ps = z['ps']
    delta = float(z['delta'])

    fig, ax = plt.subplots(figsize=(8, 5))

    for name, label, color, marker, ls in ALGOS:
        plotting.plot_with_ci(ax, ps, z[f'{name}_stop'], label=label,
                              color=color, marker=marker, ls=ls)

    log_delta = np.log(1.0 / delta)
    H_graph_med = np.median(z['H_graph'], axis=1)
    H_GF_med = np.median(z['H_GF'], axis=1)
    H_classical = float(z['H_classical'])
    ax.plot(ps, log_delta * H_GF_med,
            ':', color='tab:purple', alpha=0.7,
            label=r'$H_{\mathrm{GF}}\cdot\log(1/\delta)$')
    ax.plot(ps, log_delta * H_graph_med,
            ':', color='tab:red', alpha=0.7,
            label=r'$H_{\mathrm{graph}}\cdot\log(1/\delta)$')
    ax.axhline(log_delta * H_classical, color='gray', ls=':', alpha=0.7,
               label=r'$H_{\mathrm{classical}}\cdot\log(1/\delta)$')

    ax.set_xlabel('edge probability p')
    ax.set_ylabel('stopping time (log scale)')
    ax.set_yscale('log')
    ax.set_title('Density sweep on Erdos-Renyi graphs '
                 f'(n={int(z["n"])}, gap={float(z["gap"]):.2f})')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which='both')

    fig.tight_layout()
    fig.savefig(args.out, dpi=150, bbox_inches='tight')
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
