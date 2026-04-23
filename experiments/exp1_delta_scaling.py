"""Experiment 1 - sample-complexity scaling in 1/delta.

Validates Theorem 4.1: T ~ H_graph * log(1/delta) + polylog slack.

Default config favors correctness over speed.  Pass ``--quick`` for a
smoke test, or ``--delta-min 4`` to drop the 1e-5, 1e-6 runs if compute
is tight (see notes.md §7 "compute budget").
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.utils import instances, hardness, runners, plotting  # noqa: E402
import graph_algo  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs')
os.makedirs(OUT, exist_ok=True)


def make_factories(D, A, mu, delta, rho_lap=1.0, q=0.01):
    return {
        'ThompsonSampling': lambda: graph_algo.ThompsonSampling(
            D, A, mu, rho_lap=rho_lap, delta=delta, q=q),
        'BasicThompsonSampling': lambda: graph_algo.BasicThompsonSampling(
            mu, delta=delta, q=q),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', type=int, default=20)
    parser.add_argument('--n-jobs', type=int, default=1)
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--delta-min', type=int, default=6,
                        help='log10(1/delta_min); default 6 -> delta down to 1e-6')
    args = parser.parse_args()

    if args.quick:
        deltas = [1e-2, 1e-3]
        seeds = list(range(max(args.seeds, 3)))
    else:
        deltas = [10.0 ** -k for k in range(1, args.delta_min + 1)]
        seeds = list(range(args.seeds))

    mu, A, D = instances.sbm_standard(
        n_clusters=10, nodes_per_cluster=10, p=0.9, q=0.0,
        best_factor=1.3, seed=0)
    K = len(mu)
    H_c = hardness.classical_hardness(mu)
    H_g = hardness.graph_hardness(mu, A, D, rho=1.0)
    print(f"K={K}, H_classical={H_c:.2f}, H_graph={H_g:.2f}")

    results = {name: {} for name in ('ThompsonSampling', 'BasicThompsonSampling')}
    for delta in deltas:
        facs = make_factories(D, A, mu, delta)
        for name, fac in facs.items():
            print(f"[exp1] delta={delta:.0e} algo={name}", flush=True)
            t0 = time.time()
            runs = runners.run_many(fac, seeds, n_jobs=args.n_jobs,
                                    max_steps=1_000_000)
            print(f"       took {time.time()-t0:.1f}s, "
                  f"t_med={np.median([r['stopping_time'] for r in runs])}",
                  flush=True)
            results[name][delta] = runs

    # Save raw
    np.savez(os.path.join(OUT, 'exp1_results.npz'),
             deltas=np.array(deltas),
             seeds=np.array(seeds),
             **{f"{name}_stopping_times":
                np.array([[r['stopping_time'] for r in results[name][d]]
                          for d in deltas])
                for name in results},
             **{f"{name}_correct":
                np.array([[r['correct'] for r in results[name][d]]
                          for d in deltas])
                for name in results},
             H_classical=H_c, H_graph=H_g)

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    x = np.log(1.0 / np.array(deltas))
    for name in results:
        runs = np.array([[r['stopping_time'] for r in results[name][d]]
                         for d in deltas])
        style = plotting.style_for(name)
        plotting.plot_with_ci(ax1, x, runs, label=name, **style)
    # reference lines
    ax1.plot(x, H_c * x, '--', color='gray', alpha=0.7,
             label=f'H_classical·log(1/δ) = {H_c:.1f}·log(1/δ)')
    ax1.plot(x, H_g * x, ':', color='black', alpha=0.7,
             label=f'H_graph·log(1/δ) = {H_g:.1f}·log(1/δ)')
    ax1.set_xlabel('log(1/δ)')
    ax1.set_ylabel('stopping time')
    ax1.set_title('Sample complexity vs log(1/δ)')
    ax1.legend(loc='upper left', fontsize=8)
    ax1.grid(alpha=0.3)

    # log-log
    for name in results:
        runs = np.array([[r['stopping_time'] for r in results[name][d]]
                         for d in deltas])
        med = np.median(runs, axis=1)
        style = plotting.style_for(name)
        ax2.loglog(1.0 / np.array(deltas), med, marker=style['marker'],
                   color=style['color'], label=name)
    ax2.set_xlabel('1/δ')
    ax2.set_ylabel('median stopping time')
    ax2.set_title('Log-log check')
    ax2.legend()
    ax2.grid(True, which='both', alpha=0.3)
    fig.tight_layout()
    out_png = os.path.join(OUT, 'exp1_delta_scaling.png')
    fig.savefig(out_png, dpi=150, bbox_inches='tight')
    print(f"Saved {out_png}")

    # Regression slope
    print("\nLinear fits  T = a·log(1/δ) + b:")
    summary = []
    for name in results:
        runs = np.array([[r['stopping_time'] for r in results[name][d]]
                         for d in deltas])
        med = np.median(runs, axis=1)
        a, b = np.polyfit(x, med, 1)
        print(f"  {name}: slope={a:.2f}  intercept={b:.2f}")
        summary.append((name, a, b))

    # Acceptance checks
    t_med_per_algo = {
        name: np.median(np.array([[r['stopping_time'] for r in results[name][d]]
                                  for d in deltas]), axis=1)
        for name in results
    }
    slopes = {name: np.polyfit(x, t_med_per_algo[name], 1)[0]
              for name in t_med_per_algo}
    checks = []
    checks.append(('TS slope < Basic TS slope',
                   slopes['ThompsonSampling'] < slopes['BasicThompsonSampling']))
    checks.append(('TS slope within 2x theoretical H_graph',
                   0.5 * H_g <= slopes['ThompsonSampling'] <= 2.0 * H_g))
    err_rates = {
        name: np.mean([[not r['correct'] for r in results[name][d]]
                       for d in deltas])
        for name in results
    }
    for name, r in err_rates.items():
        checks.append((f'{name} error rate <= max(deltas)',
                       r <= max(deltas) * 5))
    print("\nAcceptance:")
    for ck, ok in checks:
        print(f"  [{'x' if ok else ' '}] {ck}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
