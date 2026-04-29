"""main_sanity — K=20 spot-check before committing to the full K sweep.

Decision point:
  * If TS-Explore at K=20 already takes >> H_graph * log(1/delta) (say, 50x
    or more), the algorithmic story is the bottleneck and we should attack
    Algorithm-1 before burning compute on K=400.
  * If the ratio is within ~10x and Basic TS on the same instance already
    looks materially worse, proceed to main_1.py with the K-sweep.

5 seeds is enough to rule out wildly outlier behaviour without taking long.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.utils import instances, hardness, runners  # noqa: E402
import graph_algo  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--K', type=int, default=20)
    parser.add_argument('--seeds', type=int, default=5)
    parser.add_argument('--q', type=float, default=0.1)
    parser.add_argument('--max-steps', type=int, default=200_000)
    args = parser.parse_args()

    delta = 1e-3
    log_inv_delta = np.log(1.0 / delta)

    mu, A, D = instances.union_of_cliques_with_challenger(args.K)
    H_classical = hardness.classical_hardness(mu)
    H_graph = hardness.graph_hardness(mu, A, D, rho=1.0)
    print(f"# K={args.K}, delta={delta}, q={args.q}, seeds={args.seeds}")
    print(f"#   H_classical = {H_classical:.2f}")
    print(f"#   H_graph     = {H_graph:.2f}")
    print(f"#   T_TS_pred  ~ H_graph * log(1/delta)     = {H_graph*log_inv_delta:.0f}")
    print(f"#   T_Basic_pred ~ H_classical * log(1/delta) = {H_classical*log_inv_delta:.0f}")
    print()

    seeds = list(range(args.seeds))

    factories = {
        'TS-Explore (graph, rho=1)': lambda: graph_algo.ThompsonSampling(
            D, A, mu, rho_lap=1.0, delta=delta, q=args.q),
        'Basic TS (no graph)': lambda: graph_algo.BasicThompsonSampling(
            mu, delta=delta, q=args.q),
        'NoGraphAlgo (UCB)': lambda: graph_algo.NoGraphAlgo(D, A, mu),
        'MaxVarianceArmAlgo (UCB+graph)': lambda: graph_algo.MaxVarianceArmAlgo(
            D, A, mu, rho_lap=1.0),
    }

    results = {}
    for name, fac in factories.items():
        print(f"  running {name} ...", flush=True, end='')
        t0 = time.time()
        runs = runners.run_many(fac, seeds, n_jobs=1, max_steps=args.max_steps)
        ts = np.array([r['stopping_time'] for r in runs], dtype=float)
        correct = np.array([r['correct'] for r in runs], dtype=bool)
        results[name] = ts
        print(f" done in {time.time()-t0:.1f}s   t_med={np.median(ts):.0f}  "
              f"t_iqr=[{np.percentile(ts,25):.0f}, {np.percentile(ts,75):.0f}]  "
              f"correct={correct.mean():.0%}", flush=True)

    print("\n# Summary")
    print(f"{'algorithm':>32} {'t_med':>8} {'t_med / H_graph*log(1/d)':>24}")
    base = H_graph * log_inv_delta
    for name, ts in results.items():
        med = np.median(ts)
        print(f"{name:>32} {med:>8.0f} {med/base:>24.2f}")

    # Decision rules
    t_ts = np.median(results['TS-Explore (graph, rho=1)'])
    t_basic = np.median(results['Basic TS (no graph)'])
    print("\n# Decision")
    print(f"  T_TS / (H_graph * log 1/d)  = {t_ts/base:.2f}")
    print(f"  T_Basic / T_TS              = {t_basic/t_ts:.2f}")
    if t_ts / base < 50 and t_basic / t_ts > 1.2:
        print("  --> proceed to main_1.py K-sweep")
    elif t_ts / base >= 50:
        print("  --> TS-Explore overshoot is too large; attack the algorithm")
    else:
        print("  --> Basic TS already comparable to TS at K=20; design too easy")


if __name__ == "__main__":
    main()
