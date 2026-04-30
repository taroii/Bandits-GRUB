"""fb_1 — graph-feedback density sweep on Erdos-Renyi (runner).

Targets ``thm:correct-fb`` and ``thm:main-fb``. Sweeps the edge density
``p`` of an Erdos-Renyi graph and tracks the empirical stopping time of
four algorithms against the analytical hardness ``H_GF``.

Algorithms compared (all with delta=1e-3):
  * TS-Explore-GF   - proposed graph-feedback Thompson sampler.
  * UCB-N           - pure-exploration variant of Caron et al. (2012).
                      Side-observation estimator + UCB-LCB elimination
                      with a max-confidence-width pull rule.
  * TS-Explore      - graph-smooth Thompson sampler at rho=1.
  * Basic TS        - empirical-mean Thompson sampler, no graph.

Each density is averaged over ``--seeds`` random graphs (one fresh ER
realization per seed) so the IQR shading captures both graph and
algorithm randomness.  Basic TS depends only on ``mu`` (constant across
densities) and is run once with the same seeds and broadcast.

Saves all raw results to ``experiments/outputs/fb_1_results.npz``.
Plotting lives in ``fb_1_plot.py``.
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

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs')
os.makedirs(OUT, exist_ok=True)


# Algorithm names used as keys in the saved npz.
GRAPH_ALGOS = ['TS-Explore', 'TS-Explore-GF', 'UCB-N']
ALL_ALGOS = GRAPH_ALGOS + ['Basic TS']


def build_factory(name, D, A, mu, delta, q):
    if name == 'TS-Explore':
        return lambda: graph_algo.ThompsonSampling(
            D=D, A=A, mu=mu, rho_lap=1.0, delta=delta, q=q)
    if name == 'TS-Explore-GF':
        return lambda: graph_algo.GraphFeedbackTS(
            D=D, A=A, mu=mu, delta=delta, q=q)
    if name == 'UCB-N':
        return lambda: graph_algo.UCB_N(
            D=D, A=A, mu=mu, delta=delta)
    if name == 'Basic TS':
        return lambda: graph_algo.BasicThompsonSampling(
            mu=mu, delta=delta, q=q)
    raise ValueError(name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', type=int, default=10)
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--q', type=float, default=0.1,
                        help="TS tail quantile; q=0.1 is the loosest stopping rule")
    parser.add_argument('--n', type=int, default=20,
                        help="ER graph size")
    parser.add_argument('--gap', type=float, default=0.3,
                        help="suboptimality gap for every non-optimal arm")
    parser.add_argument('--max-steps', type=int, default=300_000)
    args = parser.parse_args()

    if args.quick:
        ps = [0.1, 0.5, 1.0]
        seeds = list(range(max(args.seeds, 3)))
    else:
        ps = [0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
        seeds = list(range(args.seeds))

    delta = 1e-3
    n = args.n
    gap = args.gap

    # ----------------------------------------------------------------
    # State
    # ----------------------------------------------------------------
    stop_times = {a: np.full((len(ps), len(seeds)), np.nan) for a in ALL_ALGOS}
    correct = {a: np.zeros((len(ps), len(seeds)), dtype=bool) for a in ALL_ALGOS}
    H_graph = np.full((len(ps), len(seeds)), np.nan)
    H_GF = np.full((len(ps), len(seeds)), np.nan)
    # H_classical depends only on mu (constant across p and seed), so
    # one scalar suffices.
    mu_const, _, _ = instances.erdos_renyi(n=n, p=0.5, gap=gap, seed=0)
    H_classical = float(hardness.classical_hardness(mu_const))

    # ----------------------------------------------------------------
    # Basic TS broadcast (depends only on mu, not on graph).
    # ----------------------------------------------------------------
    print("[fb_1] running Basic TS once (broadcast across all p)...", flush=True)
    fac_basic = build_factory('Basic TS', None, None, mu_const, delta, args.q)
    t0 = time.time()
    basic_runs = runners.run_many(fac_basic, seeds, n_jobs=1,
                                  max_steps=args.max_steps,
                                  record_elimination=False, progress=False)
    for si, r in enumerate(basic_runs):
        stop_times['Basic TS'][:, si] = r['stopping_time']
        correct['Basic TS'][:, si] = r['correct']
    print(f"  Basic TS: t_med="
          f"{np.median([r['stopping_time'] for r in basic_runs]):.0f} "
          f"({time.time()-t0:.1f}s)", flush=True)

    # ----------------------------------------------------------------
    # (p, seed) sweep for graph-dependent algorithms.
    # Each seed picks both a fresh ER graph and the algorithm RNG.
    # ----------------------------------------------------------------
    for pi, p in enumerate(ps):
        print(f"\n=== p={p} ===", flush=True)
        for si, k in enumerate(seeds):
            mu, A, D = instances.erdos_renyi(n=n, p=p, gap=gap, seed=k)
            H_graph[pi, si] = hardness.graph_hardness(mu, A, D, rho=1.0)
            H_GF[pi, si] = hardness.graph_feedback_hardness(mu, A)
            for name in GRAPH_ALGOS:
                fac = build_factory(name, D, A, mu, delta, args.q)
                out = runners.run_algorithm(
                    fac, seed=k, max_steps=args.max_steps,
                    record_elimination=False)
                stop_times[name][pi, si] = out['stopping_time']
                correct[name][pi, si] = out['correct']
        # Per-density summary across seeds.
        print(f"  H_graph(median)={np.median(H_graph[pi]):.1f}  "
              f"H_GF(median)={np.median(H_GF[pi]):.1f}", flush=True)
        for name in GRAPH_ALGOS:
            ts = stop_times[name][pi, :]
            print(f"  {name:14s} t_med={np.median(ts):8.0f}  "
                  f"IQR=[{np.percentile(ts,25):.0f}, {np.percentile(ts,75):.0f}]  "
                  f"correct={correct[name][pi].mean()*100:.0f}%", flush=True)

    # ----------------------------------------------------------------
    # Save results
    # ----------------------------------------------------------------
    out_npz = os.path.join(OUT, 'fb_1_results.npz')
    np.savez(
        out_npz,
        ps=np.array(ps),
        seeds=np.array(seeds),
        n=n, gap=gap, delta=delta, q=args.q,
        H_classical=H_classical,
        H_graph=H_graph,
        H_GF=H_GF,
        **{f'{a}_stop': stop_times[a] for a in ALL_ALGOS},
        **{f'{a}_correct': correct[a].astype(int) for a in ALL_ALGOS},
    )
    print(f"\nSaved {out_npz}")
    print("Run experiments/fb_1_plot.py to render the figure.")

    # ----------------------------------------------------------------
    # Acceptance summary
    # ----------------------------------------------------------------
    med = {name: np.median(stop_times[name], axis=1) for name in ALL_ALGOS}
    ratio_clique = med['UCB-N'][-1] / max(med['TS-Explore-GF'][-1], 1.0)
    h_gf_shrink = np.median(H_GF[0]) / max(np.median(H_GF[-1]), 1.0)
    gfts_shrink = med['TS-Explore-GF'][0] / max(med['TS-Explore-GF'][-1], 1.0)
    print("\n# Acceptance")
    print(f"  H_GF shrinks p=0.05 -> p=1: {h_gf_shrink:.2f}x")
    print(f"  TS-Explore-GF shrinks:      {gfts_shrink:.2f}x")
    print(f"  UCB-N / TS-Explore-GF at p=1: {ratio_clique:.2f}x")
    return 0


if __name__ == "__main__":
    sys.exit(main())
