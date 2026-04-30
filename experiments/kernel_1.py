"""kernel_1 — combinatorial vs normalized Laplacian rho-sweep (runner).

Targets ``thm:kernel-ts``. On a Barabasi-Albert instance with uniform
suboptimality gap and the optimal arm pinned to the highest-degree hub
(``instances.ba_hub_optimal``), sweeps the regularization weight ``rho``
and tracks the empirical stopping time of TS-Explore under two graph
kernels: combinatorial L_G and normalized K_G.

Hypothesis: at rho=1 both kernels are dominated by direct counts and so
empirically equivalent, but as rho grows V_t becomes graph-dominated and
L_G's degree-weighted smoothing biases the optimal arm's mu_hat down
toward its (mostly suboptimal) neighborhood, shrinking the apparent gap.
K_G's normalization keeps the hub balanced, so the optimal arm's signal
is preserved and the agreement-stopping rule fires faster.  This is the
empirical mechanism behind Section "General PSD Graph Kernels".

Each seed draws a fresh BA realization (with the optimal arm pinned to
the top-degree node).  Basic TS depends only on ``mu`` and is run once
with the same seeds, broadcast across ``rhos``.

Saves all raw results to ``experiments/outputs/kernel_1_results.npz``,
checkpointed after each (rho, kernel) cell.  Plotting lives in
``kernel_1_plot.py``.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.utils import instances, runners  # noqa: E402
import graph_algo  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs')
os.makedirs(OUT, exist_ok=True)


GRAPH_ALGOS = ['TS-L_G', 'TS-K_G']
ALL_ALGOS = GRAPH_ALGOS + ['Basic TS']
KERNEL_OF = {'TS-L_G': 'combinatorial', 'TS-K_G': 'normalized'}


def build_factory(name, D, A, mu, delta, q, rho, rho_diag):
    if name in GRAPH_ALGOS:
        kernel = KERNEL_OF[name]
        return lambda: graph_algo.ThompsonSampling(
            D=D, A=A, mu=mu, rho_lap=rho, delta=delta, q=q,
            kernel=kernel, rho_diag=rho_diag)
    if name == 'Basic TS':
        return lambda: graph_algo.BasicThompsonSampling(
            mu=mu, delta=delta, q=q)
    raise ValueError(name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', type=int, default=10)
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--q', type=float, default=0.1)
    parser.add_argument('--n', type=int, default=50,
                        help="BA graph size")
    parser.add_argument('--m', type=int, default=2,
                        help="BA preferential-attachment parameter")
    parser.add_argument('--gap', type=float, default=0.3,
                        help="uniform suboptimality gap; optimal arm has "
                             "mu=1.0 and every other arm has mu=1-gap")
    parser.add_argument('--rhos', type=float, nargs='+',
                        default=[1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0])
    parser.add_argument('--max-steps', type=int, default=1_000_000)
    parser.add_argument('--fresh', action='store_true',
                        help="ignore any existing checkpoint and start over")
    args = parser.parse_args()

    if args.quick:
        rhos = [1.0, 10.0, 100.0]
        seeds = list(range(max(args.seeds, 3)))
    else:
        rhos = list(args.rhos)
        seeds = list(range(args.seeds))

    delta = 1e-3
    n, m = args.n, args.m
    gap = args.gap
    out_npz = os.path.join(OUT, 'kernel_1_results.npz')
    INSTANCE_TAG = 'ba_hub_optimal'

    # ----------------------------------------------------------------
    # State
    # ----------------------------------------------------------------
    stop_times = {a: np.full((len(rhos), len(seeds)), np.nan) for a in ALL_ALGOS}
    correct = {a: np.zeros((len(rhos), len(seeds)), dtype=bool) for a in ALL_ALGOS}
    # Per-seed instance characteristics (for context / vertical lines on plot).
    eps_L = np.full(len(seeds), np.nan)   # sqrt(<mu, L_G mu>)
    eps_K = np.full(len(seeds), np.nan)   # sqrt(<mu, K_G mu>)
    deg_max = np.full(len(seeds), np.nan)
    # Per-(rho, kernel) "done" mask for resume.
    done = np.zeros((len(rhos), len(GRAPH_ALGOS)), dtype=bool)
    basic_done = False

    def save_checkpoint():
        kwargs = dict(
            rhos=np.array(rhos),
            seeds=np.array(seeds),
            n=n, m=m, delta=delta, q=args.q, gap=gap,
            instance=np.array(INSTANCE_TAG),
            eps_L=eps_L, eps_K=eps_K, deg_max=deg_max,
            done=done.astype(int), basic_done=int(basic_done),
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
            prev_instance = str(z['instance']) if 'instance' in z.files else '<unknown>'
            if (prev_rhos == rhos and prev_seeds == seeds
                    and prev_instance == INSTANCE_TAG):
                done = z['done'].astype(bool)
                basic_done = bool(z['basic_done'])
                eps_L = z['eps_L']
                eps_K = z['eps_K']
                deg_max = z['deg_max']
                for a in ALL_ALGOS:
                    stop_times[a] = z[f'{a}_stop']
                    correct[a] = z[f'{a}_correct'].astype(bool)
                n_done = int(done.sum())
                total = len(rhos) * len(GRAPH_ALGOS)
                print(f"[resume] loaded {n_done}/{total} cells, "
                      f"basic_done={basic_done}", flush=True)
            else:
                print(f"[resume] checkpoint mismatch "
                      f"(prev_instance={prev_instance!r}, want "
                      f"{INSTANCE_TAG!r}); ignoring "
                      f"(use --fresh to silence)", flush=True)
        except Exception as e:
            print(f"[resume] failed to load {out_npz}: {e}", flush=True)

    # ----------------------------------------------------------------
    # Per-seed instance characterization (cheap; do once).
    # Also Basic TS broadcast (depends only on mu).
    # ----------------------------------------------------------------
    print("[kernel_1] characterizing instances and running Basic TS...",
          flush=True)
    mu_const = None
    for si, k in enumerate(seeds):
        mu, A, D = instances.ba_hub_optimal(n=n, m=m, gap=gap, seed=k)
        if mu_const is None:
            mu_const = mu  # Basic TS only needs mu (constant in design)
        L = D - A
        eps_L[si] = float(np.sqrt(max(mu @ L @ mu, 0.0)))
        # K_G = I - D^{-1/2} A D^{-1/2}
        d = np.diag(D)
        d_inv_sqrt = np.where(d > 0, 1.0 / np.sqrt(np.maximum(d, 1e-12)), 0.0)
        K = np.eye(n) - (d_inv_sqrt[:, None] * A) * d_inv_sqrt[None, :]
        eps_K[si] = float(np.sqrt(max(mu @ K @ mu, 0.0)))
        deg_max[si] = float(d.max())

    print(f"  eps_L (median): {np.nanmedian(eps_L):.3f}  "
          f"eps_K (median): {np.nanmedian(eps_K):.3f}  "
          f"deg_max (median): {np.nanmedian(deg_max):.0f}",
          flush=True)
    save_checkpoint()

    if not basic_done:
        fac_basic = build_factory('Basic TS', None, None, mu_const, delta,
                                  args.q, rho=0.0, rho_diag=1e-4)
        t0 = time.time()
        basic_runs = runners.run_many(fac_basic, seeds, n_jobs=1,
                                      max_steps=args.max_steps,
                                      record_elimination=False, progress=False)
        for si, r in enumerate(basic_runs):
            stop_times['Basic TS'][:, si] = r['stopping_time']
            correct['Basic TS'][:, si] = r['correct']
        basic_done = True
        save_checkpoint()
        print(f"  Basic TS: t_med="
              f"{np.median([r['stopping_time'] for r in basic_runs]):.0f} "
              f"({time.time()-t0:.1f}s)", flush=True)
    else:
        print("  Basic TS [resumed]", flush=True)

    # ----------------------------------------------------------------
    # Sweep over (rho, kernel).  Inner loop = seeds (fresh graph per seed).
    # ----------------------------------------------------------------
    for ri, rho in enumerate(rhos):
        # Scale rho_diag with rho to keep V_0 well-conditioned at high rho.
        rho_diag = max(1e-4, 1e-6 * rho)
        print(f"\n=== rho={rho}  rho_diag={rho_diag:.1e} ===", flush=True)
        for ai, name in enumerate(GRAPH_ALGOS):
            if done[ri, ai]:
                ts = stop_times[name][ri, :]
                print(f"  {name:8s} [resumed] t_med={np.median(ts):.0f}",
                      flush=True)
                continue
            t0 = time.time()
            for si, k in enumerate(seeds):
                mu, A, D = instances.ba_hub_optimal(n=n, m=m, gap=gap, seed=k)
                fac = build_factory(name, D, A, mu, delta, args.q,
                                    rho=rho, rho_diag=rho_diag)
                out = runners.run_algorithm(
                    fac, seed=k, max_steps=args.max_steps,
                    record_elimination=False)
                stop_times[name][ri, si] = out['stopping_time']
                correct[name][ri, si] = out['correct']
            done[ri, ai] = True
            save_checkpoint()
            ts = stop_times[name][ri, :]
            cor = correct[name][ri, :]
            elapsed = time.time() - t0
            print(f"  {name:8s} t_med={np.median(ts):8.0f}  "
                  f"IQR=[{np.percentile(ts,25):.0f}, {np.percentile(ts,75):.0f}]  "
                  f"correct={cor.mean()*100:.0f}%  ({elapsed:.0f}s)",
                  flush=True)

    # ----------------------------------------------------------------
    # Acceptance summary
    # ----------------------------------------------------------------
    med_L = np.median(stop_times['TS-L_G'], axis=1)
    med_K = np.median(stop_times['TS-K_G'], axis=1)
    print(f"\nSaved {out_npz}")
    print("Run experiments/kernel_1_plot.py to render the figure.")
    print("\n# Acceptance")
    for ri, rho in enumerate(rhos):
        ratio = med_L[ri] / max(med_K[ri], 1.0)
        print(f"  rho={rho:6.1f}  L_G={med_L[ri]:>8.0f}  K_G={med_K[ri]:>8.0f}  "
              f"L_G/K_G={ratio:.2f}x")
    best_L = float(np.nanmin(med_L))
    best_K = float(np.nanmin(med_K))
    rho_L_star = rhos[int(np.nanargmin(med_L))]
    rho_K_star = rhos[int(np.nanargmin(med_K))]
    print(f"\n  L_G U-bottom: T={best_L:.0f} at rho={rho_L_star}")
    print(f"  K_G U-bottom: T={best_K:.0f} at rho={rho_K_star}")
    print(f"  Best K_G / Best L_G: {best_K / max(best_L, 1.0):.2f}x  "
          f"({'K_G wins' if best_K < best_L else 'L_G wins or tie'})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
