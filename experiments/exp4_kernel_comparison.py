"""Experiment 4 - Laplacian vs normalized Laplacian on heterogeneous-degree graphs."""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.utils import instances, runners, plotting  # noqa: E402
import graph_algo  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs')
os.makedirs(OUT, exist_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', type=int, default=20)
    parser.add_argument('--n-jobs', type=int, default=1)
    parser.add_argument('--quick', action='store_true')
    args = parser.parse_args()

    delta = 1e-3
    seeds = list(range(args.seeds if not args.quick else 3))

    datasets = {
        'BA_n100_m2': instances.barabasi_albert(n=100, m=2, gap=0.3, seed=0),
        'SBM_10x10_p0.9': instances.sbm_standard(
            n_clusters=10, nodes_per_cluster=10, p=0.9, best_factor=1.3, seed=0),
    }
    kernels = ['combinatorial', 'normalized']

    stop_times = {(ds, k): np.zeros(len(seeds)) for ds in datasets for k in kernels}
    pull_counts = {(ds, k): [] for ds in datasets for k in kernels}

    for ds_name, (mu, A, D) in datasets.items():
        print(f"[exp4] instance={ds_name}, K={len(mu)}", flush=True)
        for kernel in kernels:
            fac = lambda mu=mu, A=A, D=D, kernel=kernel: graph_algo.ThompsonSampling(
                D, A, mu, rho_lap=1.0, delta=delta, q=0.01,
                kernel=kernel)
            t0 = time.time()
            runs = runners.run_many(fac, seeds, n_jobs=args.n_jobs,
                                    max_steps=500_000)
            print(f"    {kernel}: t_med={np.median([r['stopping_time'] for r in runs]):.0f} "
                  f"({time.time()-t0:.1f}s)", flush=True)
            for si, r in enumerate(runs):
                stop_times[(ds_name, kernel)][si] = r['stopping_time']
                pull_counts[(ds_name, kernel)].append(r['pull_counts'])

    np.savez(os.path.join(OUT, 'exp4_results.npz'),
             **{f'{ds}_{k}_stop': stop_times[(ds, k)]
                for ds in datasets for k in kernels},
             **{f'{ds}_{k}_pulls': np.array(pull_counts[(ds, k)])
                for ds in datasets for k in kernels})

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Bar chart
    xpos = np.arange(len(datasets))
    w = 0.35
    for i, kernel in enumerate(kernels):
        med = np.array([np.median(stop_times[(ds, kernel)]) for ds in datasets])
        p25 = np.array([np.percentile(stop_times[(ds, kernel)], 25) for ds in datasets])
        p75 = np.array([np.percentile(stop_times[(ds, kernel)], 75) for ds in datasets])
        axes[0].bar(xpos + (i - 0.5) * w, med, width=w, label=kernel,
                    yerr=[med - p25, p75 - med], capsize=4)
    axes[0].set_xticks(xpos)
    axes[0].set_xticklabels(list(datasets.keys()), rotation=10)
    axes[0].set_ylabel('stopping time (median, IQR)')
    axes[0].set_title('Stopping time by kernel')
    axes[0].legend()
    axes[0].grid(axis='y', alpha=0.3)

    # Pull-count distribution on BA
    ba_combin = np.concatenate(pull_counts[('BA_n100_m2', 'combinatorial')])
    ba_norm = np.concatenate(pull_counts[('BA_n100_m2', 'normalized')])
    axes[1].hist(ba_combin, bins=40, alpha=0.5, label='combinatorial', density=True)
    axes[1].hist(ba_norm, bins=40, alpha=0.5, label='normalized', density=True)
    axes[1].set_xlabel('pulls per arm (BA instance)')
    axes[1].set_ylabel('density')
    axes[1].set_title('Pull-count distribution (BA)')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    out_png = os.path.join(OUT, 'exp4_kernel_comparison.png')
    fig.savefig(out_png, dpi=150, bbox_inches='tight')
    print(f"Saved {out_png}")

    # Acceptance
    ba_c = np.median(stop_times[('BA_n100_m2', 'combinatorial')])
    ba_n = np.median(stop_times[('BA_n100_m2', 'normalized')])
    sbm_c = np.median(stop_times[('SBM_10x10_p0.9', 'combinatorial')])
    sbm_n = np.median(stop_times[('SBM_10x10_p0.9', 'normalized')])
    print("\nAcceptance:")
    print(f"  BA: combin={ba_c:.0f}, norm={ba_n:.0f} -> "
          f"[{'x' if ba_n <= ba_c else ' '}] normalized <= combinatorial")
    print(f"  SBM: combin={sbm_c:.0f}, norm={sbm_n:.0f} -> "
          f"[{'x' if abs(sbm_n - sbm_c)/max(sbm_c, 1) <= 0.2 else ' '}] within 20%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
