"""movielens_1 -- real-graph hero experiment (rho-sweep).

Targets ``thm:main-graph`` on a real-world graph: the K=20 most-rated
MovieLens-100K movies as arms, with item-item adjusted-cosine similarity
sparsified to a top-k mutual-neighbor graph.  ``mu_i`` is the empirical
mean rating; rewards are sampled as N(mu_i, sigma=1).

We sweep the regularization weight rho for the graph-aware algorithms
(TS-Explore and GRUB) and broadcast Basic TS across rho as a no-graph
reference.  Hypothesis: TS-Explore exhibits a U-shape over rho, with
its minimum stopping time below Basic TS at the rho prescribed by the
analysis (rho^*(eps) ~ sigma_0 sqrt(L_1) / eps, here ~10-30).

Saves all raw results to ``experiments/outputs/movielens_1_results.npz``.
Plotting lives in ``movielens_1_plot.py``.

Data: ml-100k.zip is auto-downloaded on first run to <repo>/data/ml-100k/.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.utils import instances, hardness, runners, movielens  # noqa: E402
import graph_algo  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs')
os.makedirs(OUT, exist_ok=True)


GRAPH_ALGOS = ['TS-Explore', 'GRUB']
ALL_ALGOS = GRAPH_ALGOS + ['Basic TS']
INSTANCE_TAG = 'movielens_top_k'


def build_factory(name, D, A, mu, delta, q, rho):
    if name == 'TS-Explore':
        return lambda: graph_algo.ThompsonSampling(
            D=D, A=A, mu=mu, rho_lap=rho, delta=delta, q=q)
    if name == 'GRUB':
        return lambda: graph_algo.MaxVarianceArmAlgo(
            D=D, A=A, mu=mu, rho_lap=rho, delta=delta)
    if name == 'Basic TS':
        return lambda: graph_algo.BasicThompsonSampling(
            mu=mu, delta=delta, q=q)
    raise ValueError(name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--K', type=int, default=20,
                        help="number of arms (top-K by rating count). "
                             "K > 20 makes BAI intractable on this dataset "
                             "(gap_min collapses below 0.02).")
    parser.add_argument('--top-k-neighbors', type=int, default=5)
    parser.add_argument('--min-common', type=int, default=5)
    parser.add_argument('--rhos', type=float, nargs='+',
                        default=[1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0])
    parser.add_argument('--seeds', type=int, default=20)
    parser.add_argument('--n-jobs', type=int, default=1)
    parser.add_argument('--q', type=float, default=0.1)
    parser.add_argument('--max-steps', type=int, default=10_000_000,
                        help="hard cap on steps per run; the smallest gap "
                             "in MovieLens top-20 is ~0.07 so Basic TS "
                             "needs O(2M) pulls in worst case")
    parser.add_argument('--quick', action='store_true',
                        help="rhos=[1, 30], seeds=3, max-steps=2_000_000")
    parser.add_argument('--algos', type=str, nargs='+', default=ALL_ALGOS,
                        choices=ALL_ALGOS,
                        help="subset of algorithms to run")
    parser.add_argument('--fresh', action='store_true',
                        help="ignore any existing checkpoint and start over")
    args = parser.parse_args()

    if args.quick:
        rhos = [1.0, 30.0]
        seeds = list(range(max(args.seeds, 3) if args.seeds < 3 else 3))
        args.max_steps = min(args.max_steps, 2_000_000)
    else:
        rhos = list(args.rhos)
        seeds = list(range(args.seeds))

    delta = 1e-3
    out_npz = os.path.join(OUT, 'movielens_1_results.npz')
    K = args.K

    # ----------------------------------------------------------------
    # Build instance once (deterministic) and report.
    # ----------------------------------------------------------------
    print(f"[movielens_1] loading K={K} top-rated movies...", flush=True)
    mu, A, D, meta = movielens.build_instance(
        K=K, top_k_neighbors=args.top_k_neighbors,
        min_common=args.min_common, return_meta=True,
    )
    L = D - A
    eps2 = float(mu @ L @ mu)
    eps = float(np.sqrt(max(eps2, 0.0)))
    a_star = int(np.argmax(mu))
    Delta_pos = (mu[a_star] - mu)[mu < mu[a_star]]
    H_cls = hardness.classical_hardness(mu)
    n_edges = int(A.sum() / 2)
    deg = np.diag(D)
    print(f"  best arm: '{meta['titles'][a_star]}'  mu={mu[a_star]:.3f}",
          flush=True)
    print(f"  K={K}, edges={n_edges}, avg_deg={2*n_edges/K:.1f}, "
          f"max_deg={int(deg.max())}", flush=True)
    print(f"  gap_min={Delta_pos.min():.4f}  "
          f"gap_med={np.median(Delta_pos):.3f}  "
          f"gap_max={Delta_pos.max():.3f}", flush=True)
    print(f"  epsilon={eps:.3f}  H_classical={H_cls:.1f}", flush=True)

    algos_to_run = [a for a in ALL_ALGOS if a in args.algos]
    graph_algos_to_run = [a for a in GRAPH_ALGOS if a in algos_to_run]
    basic_in = 'Basic TS' in algos_to_run

    # ----------------------------------------------------------------
    # State (with checkpoint resume)
    # ----------------------------------------------------------------
    stop_times = {a: np.full((len(rhos), len(seeds)), np.nan) for a in ALL_ALGOS}
    correct = {a: np.zeros((len(rhos), len(seeds)), dtype=bool) for a in ALL_ALGOS}
    H_graph_per_rho = np.full(len(rhos), np.nan)
    done = np.zeros((len(rhos), len(GRAPH_ALGOS)), dtype=bool)
    basic_done = False

    def save_checkpoint():
        kwargs = dict(
            rhos=np.array(rhos),
            seeds=np.array(seeds),
            K=int(K),
            top_k_neighbors=int(args.top_k_neighbors),
            min_common=int(args.min_common),
            n_edges=int(n_edges),
            eps=float(eps),
            H_classical=float(H_cls),
            H_graph=H_graph_per_rho,
            mu=mu,
            instance=np.array(INSTANCE_TAG),
            delta=delta,
            q=args.q,
            done=done.astype(int),
            basic_done=int(basic_done),
        )
        for a in ALL_ALGOS:
            kwargs[f'{a}_stop'] = stop_times[a]
            kwargs[f'{a}_correct'] = correct[a].astype(int)
        tmp = out_npz + '.tmp.npz'
        np.savez(tmp, **kwargs)
        os.replace(tmp, out_npz)

    if os.path.exists(out_npz) and not args.fresh:
        try:
            z = np.load(out_npz, allow_pickle=False)
            prev_rhos = list(z['rhos'].tolist())
            prev_seeds = list(z['seeds'].tolist())
            prev_instance = (str(z['instance'])
                             if 'instance' in z.files else '<unknown>')
            prev_K = int(z['K']) if 'K' in z.files else -1
            prev_topk = int(z['top_k_neighbors']) if 'top_k_neighbors' in z.files else -1
            if (prev_rhos == rhos and prev_seeds == seeds
                    and prev_instance == INSTANCE_TAG
                    and prev_K == int(K)
                    and prev_topk == int(args.top_k_neighbors)):
                done = z['done'].astype(bool)
                basic_done = bool(z['basic_done'])
                if 'H_graph' in z.files:
                    H_graph_per_rho = z['H_graph']
                for a in ALL_ALGOS:
                    stop_times[a] = z[f'{a}_stop']
                    correct[a] = z[f'{a}_correct'].astype(bool)
                n_done = int(done.sum())
                total = len(rhos) * len(GRAPH_ALGOS)
                print(f"[resume] loaded {n_done}/{total} graph-cells, "
                      f"basic_done={basic_done}", flush=True)
            else:
                print(f"[resume] checkpoint mismatch "
                      f"(prev_K={prev_K}, prev_topk={prev_topk}, "
                      f"prev_instance={prev_instance!r}); ignoring "
                      f"(use --fresh to silence)", flush=True)
        except Exception as e:
            print(f"[resume] failed to load {out_npz}: {e}", flush=True)

    # Compute H_graph at each rho (cheap; do once, even on resume).
    for ri, rho in enumerate(rhos):
        if np.isnan(H_graph_per_rho[ri]):
            H_graph_per_rho[ri] = hardness.graph_hardness(mu, A, D, rho=rho)
    save_checkpoint()

    # ----------------------------------------------------------------
    # Basic TS (broadcast).
    # ----------------------------------------------------------------
    if basic_in and not basic_done:
        fac = build_factory('Basic TS', None, None, mu, delta, args.q, rho=0.0)
        t0 = time.time()
        runs = runners.run_many(fac, seeds, n_jobs=args.n_jobs,
                                max_steps=args.max_steps,
                                record_elimination=False, progress=False)
        ts = np.array([r['stopping_time'] for r in runs], dtype=float)
        cor = np.array([r['correct'] for r in runs], dtype=bool)
        for ri in range(len(rhos)):
            stop_times['Basic TS'][ri, :] = ts
            correct['Basic TS'][ri, :] = cor
        basic_done = True
        save_checkpoint()
        print(f"\nBasic TS (broadcast)  t_med={np.median(ts):8.0f}  "
              f"IQR=[{np.percentile(ts,25):.0f}, {np.percentile(ts,75):.0f}]  "
              f"correct={cor.mean():.0%}  ({time.time()-t0:.0f}s)", flush=True)
    elif basic_in:
        ts = stop_times['Basic TS'][0, :]
        print(f"\nBasic TS [resumed]    t_med={np.median(ts):8.0f}", flush=True)

    # ----------------------------------------------------------------
    # rho-sweep for graph algorithms.
    # ----------------------------------------------------------------
    for ri, rho in enumerate(rhos):
        H_gr = H_graph_per_rho[ri]
        print(f"\n=== rho={rho}  H_graph={H_gr:.1f}  "
              f"(H_cls/H_gr = {H_cls/max(H_gr,1e-9):.2f}x) ===", flush=True)
        for ai, name in enumerate(GRAPH_ALGOS):
            if name not in graph_algos_to_run:
                continue
            if done[ri, ai]:
                ts = stop_times[name][ri, :]
                print(f"  {name:11s} [resumed] t_med={np.median(ts):8.0f}",
                      flush=True)
                continue
            fac = build_factory(name, D, A, mu, delta, args.q, rho=rho)
            t0 = time.time()
            runs = runners.run_many(fac, seeds, n_jobs=args.n_jobs,
                                    max_steps=args.max_steps,
                                    record_elimination=False, progress=False)
            ts = np.array([r['stopping_time'] for r in runs], dtype=float)
            cor = np.array([r['correct'] for r in runs], dtype=bool)
            converged = np.array([r['converged_flag'] for r in runs],
                                 dtype=bool)
            stop_times[name][ri, :] = ts
            correct[name][ri, :] = cor
            done[ri, ai] = True
            save_checkpoint()
            n_unconv = int((~converged).sum())
            unconv_str = (f", unconverged={n_unconv}/{len(seeds)}"
                          if n_unconv else "")
            print(f"  {name:11s} t_med={np.median(ts):8.0f}  "
                  f"IQR=[{np.percentile(ts,25):.0f}, "
                  f"{np.percentile(ts,75):.0f}]  "
                  f"correct={cor.mean():.0%}{unconv_str}  "
                  f"({time.time()-t0:.0f}s)", flush=True)

    print(f"\nSaved {out_npz}")
    print("Run experiments/movielens_1_plot.py to render the figure.")

    # Acceptance summary.
    if 'TS-Explore' in algos_to_run:
        med_ts = np.median(stop_times['TS-Explore'], axis=1)
        print("\n# Acceptance")
        for ri, rho in enumerate(rhos):
            line = f"  rho={rho:6.1f}  TS-Explore={med_ts[ri]:>8.0f}"
            if basic_in:
                med_b = np.median(stop_times['Basic TS'], axis=1)[ri]
                ratio = med_b / max(med_ts[ri], 1.0)
                line += f"  Basic={med_b:>8.0f}  Basic/TS={ratio:.2f}x"
            print(line)
        if basic_in:
            best_ts = float(np.nanmin(med_ts))
            rho_star = rhos[int(np.nanargmin(med_ts))]
            med_b0 = float(np.median(stop_times['Basic TS'][0, :]))
            print(f"\n  TS-Explore U-bottom: T={best_ts:.0f} at rho={rho_star}")
            print(f"  Best speedup vs Basic TS: "
                  f"{med_b0 / max(best_ts, 1.0):.2f}x")
    return 0


if __name__ == "__main__":
    sys.exit(main())
