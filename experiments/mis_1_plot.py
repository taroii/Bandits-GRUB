"""mis_1_plot — render Figure 2 from mis_1_results.npz.

Reads ``experiments/outputs/mis_1_results.npz`` (produced by ``mis_1.py``)
and writes ``experiments/outputs/mis_1.png``.

Three panels stacked:
  * Top    — median stopping time vs log10(epsilon), with 25--75 IQR shading.
  * Middle — analytical H_epsilon (step plot).
  * Bottom — competitive-set size |H_epsilon|.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.utils import instances, plotting  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs')

# Keep these in sync with ``mis_1.py``.
ALGOS = [
    ('TS_tuned', 'TS (rho* tuned)',     '#d62728', 's', '-'),
    ('TS_rho1',  'TS (rho=1, fixed)',   '#ff9896', 'o', '--'),
    ('Basic',    'Basic TS (no graph)', '#2ca02c', '^', ':'),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=str,
                        default=os.path.join(OUT, 'mis_1_results.npz'))
    parser.add_argument('--out', type=str,
                        default=os.path.join(OUT, 'mis_1.png'))
    parser.add_argument('--instance-seed', type=int, default=0,
                        help="Seed used for the SBM instance in mis_1.py "
                             "(default 0). Must match the runner.")
    parser.add_argument('--delta', type=float, default=1e-3,
                        help="Confidence parameter used in the runner "
                             "(default 1e-3). Must match the runner.")
    args = parser.parse_args()

    if not os.path.exists(args.results):
        print(f"Error: {args.results} not found. "
              f"Run experiments/mis_1.py first.", file=sys.stderr)
        sys.exit(1)

    z = np.load(args.results, allow_pickle=False)
    eps = z['eps']
    log_eps = np.log10(eps)

    # The instance is fully determined by its seed; rebuild it here so we can
    # draw the analytical limits without serializing mu/D into the npz.
    mu, A, D = instances.sbm_phase_transition(seed=args.instance_seed)
    K = len(mu)
    a_star = int(np.argmax(mu))
    gap2_nonstar = [(mu[a_star] - mu[i]) ** 2 for i in range(K) if i != a_star]
    asym = (1.0 / min(gap2_nonstar)) * np.log(1.0 / args.delta)
    sum_limit = sum(1.0 / g for g in gap2_nonstar)
    max_limit = 1.0 / min(gap2_nonstar)

    fig, axes = plt.subplots(3, 1, figsize=(9, 11), sharex=True)

    # Panel A: stopping time
    for name, label, color, marker, ls in ALGOS:
        plotting.plot_with_ci(axes[0], log_eps, z[f'{name}_stop'],
                              label=label, color=color, marker=marker, ls=ls)
    eps_critical = z['eps_critical']
    for v in eps_critical:
        if np.isfinite(v) and v > 0:
            axes[0].axvline(np.log10(v), color='k', alpha=0.3, ls=':',
                            linewidth=1)
    axes[0].axhline(asym, color='black', ls='--', alpha=0.5,
                    label=f'max 1/Delta^2 * log(1/delta) = {asym:.0f}')
    axes[0].set_ylabel('stopping time')
    axes[0].set_yscale('log')
    axes[0].set_title('Panel A: stopping time vs log10(eps)')
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3, which='both')

    # Panel B: H_eps
    axes[1].step(log_eps, z['H_eps'], where='mid', color='tab:blue')
    axes[1].axhline(sum_limit, color='gray', ls='--', alpha=0.7,
                    label=f'sum 1/Delta^2 = {sum_limit:.1f}')
    axes[1].axhline(max_limit, color='black', ls='--', alpha=0.7,
                    label=f'max 1/Delta^2 = {max_limit:.1f}')
    axes[1].set_ylabel('H_eps')
    axes[1].set_title('Panel B: analytical H_eps')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    # Panel C: |competitive set|
    axes[2].step(log_eps, z['comp_size'], where='mid', color='tab:green')
    axes[2].set_xlabel('log10(eps)')
    axes[2].set_ylabel('|competitive set|')
    axes[2].set_title('Panel C: competitive-set size')
    axes[2].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(args.out, dpi=150, bbox_inches='tight')
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
