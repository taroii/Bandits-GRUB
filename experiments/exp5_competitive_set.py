"""Experiment 5 - per-arm pull decomposition by competitive / non-competitive."""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.utils import instances, hardness, runners  # noqa: E402
import graph_algo  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs')
os.makedirs(OUT, exist_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', type=int, default=10)
    parser.add_argument('--n-jobs', type=int, default=1)
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--q', type=float, default=0.1)
    parser.add_argument('--max-steps', type=int, default=300_000)
    args = parser.parse_args()

    delta = 1e-3
    rho_lap = 1.0
    seeds = list(range(args.seeds if not args.quick else 5))

    # Downscaled from notes.md's K=101 SBM since the full instance does not
    # converge in reasonable time (see phase0_log.md).
    mu, A, D = instances.sbm_standard(
        n_clusters=2, nodes_per_cluster=5, p=0.9, q=0.0,
        best_factor=1.3, seed=0)
    K = len(mu)

    H_idx, N_idx = hardness.competitive_set(mu, A, D, rho=rho_lap)
    a_star = int(np.argmax(mu))
    print(f"[exp5] K={K}, |competitive|={len(H_idx)}, "
          f"|non-competitive|={len(N_idx)}", flush=True)

    fac = lambda: graph_algo.ThompsonSampling(
        D, A, mu, rho_lap=rho_lap, delta=delta, q=args.q)
    t0 = time.time()
    runs = runners.run_many(fac, seeds, n_jobs=args.n_jobs,
                            max_steps=args.max_steps)
    print(f"[exp5] {len(seeds)} seeds, "
          f"t_med={np.median([r['stopping_time'] for r in runs]):.0f} "
          f"({time.time()-t0:.1f}s)", flush=True)

    pull_matrix = np.array([r['pull_counts'] for r in runs])
    pull_med = np.median(pull_matrix, axis=0)

    gaps = mu[a_star] - mu
    gap2 = gaps ** 2

    np.savez(os.path.join(OUT, 'exp5_results.npz'),
             pull_matrix=pull_matrix,
             means=mu, gaps=gaps,
             H_idx=np.array(H_idx, dtype=int),
             N_idx=np.array(N_idx, dtype=int))

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    # Panel A: bar per arm, colored by membership, sorted by gap
    order = np.argsort(-gap2)  # largest gap first... but best is arm a_star with gap=0
    colors = []
    for i in range(K):
        if i == a_star:
            colors.append('tab:green')
        elif i in H_idx:
            colors.append('tab:blue')
        elif i in N_idx:
            colors.append('tab:orange')
        else:
            colors.append('gray')
    axes[0, 0].bar(range(K), pull_med[order],
                   color=[colors[i] for i in order])
    axes[0, 0].set_xlabel('arm (sorted by gap, smallest on right)')
    axes[0, 0].set_ylabel('median pull count')
    axes[0, 0].set_title('Panel A: pull counts (blue=comp, orange=non-comp, green=best)')
    axes[0, 0].set_yscale('log')
    axes[0, 0].grid(axis='y', alpha=0.3)

    # Panel B: scatter 1/Delta^2 vs median pulls for competitive arms
    if H_idx:
        x_b = 1.0 / gap2[np.array(H_idx)]
        y_b = pull_med[np.array(H_idx)]
        axes[0, 1].scatter(x_b, y_b, color='tab:blue')
        if len(H_idx) > 1:
            r = np.corrcoef(x_b, y_b)[0, 1]
            axes[0, 1].set_title(f'Panel B: competitive arms (Pearson r = {r:.2f})')
        else:
            axes[0, 1].set_title('Panel B: competitive arms (n=1)')
    axes[0, 1].set_xlabel('1/Δ²')
    axes[0, 1].set_ylabel('median pull count')
    axes[0, 1].set_xscale('log')
    axes[0, 1].set_yscale('log')
    axes[0, 1].grid(alpha=0.3, which='both')

    # Panel C: histogram of non-competitive pull counts
    if N_idx:
        axes[1, 0].hist(pull_med[np.array(N_idx)], bins=20, color='tab:orange')
    axes[1, 0].set_xlabel('median pull count (non-competitive arms)')
    axes[1, 0].set_ylabel('count')
    axes[1, 0].set_title('Panel C: non-competitive pull distribution')
    axes[1, 0].grid(alpha=0.3)

    # Panel D: pull_count * Delta^2 (roughly constant on competitive, near 0 on non-competitive)
    prod = np.zeros(K)
    nonzero = gap2 > 0
    prod[nonzero] = pull_med[nonzero] * gap2[nonzero]
    axes[1, 1].bar(range(K), prod[order], color=[colors[i] for i in order])
    axes[1, 1].set_xlabel('arm (sorted by gap)')
    axes[1, 1].set_ylabel('pull_count · Δ²')
    axes[1, 1].set_title('Panel D: pulls weighted by Δ²')
    axes[1, 1].grid(axis='y', alpha=0.3)

    fig.tight_layout()
    out_png = os.path.join(OUT, 'exp5_competitive_set.png')
    fig.savefig(out_png, dpi=150, bbox_inches='tight')
    print(f"Saved {out_png}")

    # Acceptance
    if len(H_idx) > 1:
        x_b = 1.0 / gap2[np.array(H_idx)]
        y_b = pull_med[np.array(H_idx)]
        r = np.corrcoef(x_b, y_b)[0, 1]
        print(f"\nAcceptance:  Pearson r on competitive arms: {r:.2f} "
              f"[{'x' if r >= 0.7 else ' '}] (target >= 0.7)")
    if N_idx:
        mean_noncomp = float(pull_med[np.array(N_idx)].mean())
        print(f"  Non-competitive mean pulls: {mean_noncomp:.1f} "
              f"(should be ~O(log T), not O(1/Delta^2))")
    return 0


if __name__ == "__main__":
    sys.exit(main())
