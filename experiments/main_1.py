"""main_1 — Figure 1 experiment runner: TS-Explore vs GRUB.

Targets ``thm:main-graph``. Runs the K-sweep on
``union_of_cliques_with_challenger`` for TS-Explore (graph), Basic TS,
and GRUB (Thaker et al. 2022, = ``MaxVarianceArmAlgo`` with
``eliminate_arms`` from ``algobase.py``).  Also collects a single-seed
elimination/agreement curve at one K for the second panel of the
figure.

Saves all raw results to ``experiments/outputs/main_1_results.npz``.
Plotting lives in ``main_1_plot.py`` so figures can be iterated on
without rerunning the experiment.
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


# Top-level picklable factories (multiprocessing requires this).
class TSExploreFactory:
    def __init__(self, D, A, mu, delta, q):
        self.kw = dict(D=D, A=A, mu=mu, rho_lap=1.0, delta=delta, q=q)

    def __call__(self):
        return graph_algo.ThompsonSampling(**self.kw)


class BasicTSFactory:
    def __init__(self, mu, delta, q):
        self.kw = dict(mu=mu, delta=delta, q=q)

    def __call__(self):
        return graph_algo.BasicThompsonSampling(**self.kw)


class GRUBFactory:
    def __init__(self, D, A, mu, delta):
        self.kw = dict(D=D, A=A, mu=mu, rho_lap=1.0, delta=delta)

    def __call__(self):
        return graph_algo.MaxVarianceArmAlgo(**self.kw)


ALGOS = ['TS-Explore', 'Basic TS', 'GRUB']


def make_factory(name, D, A, mu, delta, q):
    if name == 'TS-Explore':
        return TSExploreFactory(D, A, mu, delta, q)
    if name == 'Basic TS':
        return BasicTSFactory(mu, delta, q)
    if name == 'GRUB':
        return GRUBFactory(D, A, mu, delta=delta)
    raise ValueError(name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--Ks', type=int, nargs='+',
                        default=[20, 50, 100],
                        help="K values to sweep (default: 20 50 100)")
    parser.add_argument('--seeds', type=int, default=20)
    parser.add_argument('--n-jobs', type=int, default=1)
    parser.add_argument('--q', type=float, default=0.1)
    parser.add_argument('--max-steps', type=int, default=2_000_000)
    parser.add_argument('--panel-b-K', type=int, default=100,
                        help="K for the elimination-curve panel (default 100)")
    parser.add_argument('--panel-b-seed', type=int, default=0)
    parser.add_argument('--quick', action='store_true',
                        help="K=[20,50], seeds=5")
    parser.add_argument('--fresh', action='store_true',
                        help="ignore any existing checkpoint and start over")
    args = parser.parse_args()

    if args.quick:
        Ks = [20, 50]
        seeds = list(range(5))
        args.panel_b_K = 50
    else:
        Ks = args.Ks
        seeds = list(range(args.seeds))

    delta = 1e-3
    out_npz = os.path.join(OUT, 'main_1_results.npz')

    # ------------------------------------------------------------------
    # State (with checkpoint resume)
    # ------------------------------------------------------------------
    stop_times = {a: np.full((len(Ks), len(seeds)), np.nan) for a in ALGOS}
    correct = {a: np.zeros((len(Ks), len(seeds)), dtype=bool) for a in ALGOS}
    H_classical = [None] * len(Ks)
    H_graph_vals = [None] * len(Ks)
    done = np.zeros((len(Ks), len(ALGOS)), dtype=bool)
    curves = {a: None for a in ALGOS}
    panel_b_done = False

    def save_checkpoint():
        kwargs = dict(
            Ks=np.array(Ks),
            seeds=np.array(seeds),
            H_classical=np.array(
                [v if v is not None else np.nan for v in H_classical]),
            H_graph=np.array(
                [v if v is not None else np.nan for v in H_graph_vals]),
            panel_b_K=args.panel_b_K,
            panel_b_seed=args.panel_b_seed,
            delta=delta,
            q=args.q,
            done=done,
            panel_b_done=panel_b_done,
        )
        for a in ALGOS:
            kwargs[f'{a}_stop'] = stop_times[a]
            kwargs[f'{a}_correct'] = correct[a].astype(int)
            if curves[a] is not None:
                kwargs[f'{a}_curve_t'] = np.array(
                    [p[0] for p in curves[a]], dtype=float)
                kwargs[f'{a}_curve_n'] = np.array(
                    [p[1] for p in curves[a]], dtype=float)
            else:
                kwargs[f'{a}_curve_t'] = np.zeros(0, dtype=float)
                kwargs[f'{a}_curve_n'] = np.zeros(0, dtype=float)
        tmp = out_npz + '.tmp.npz'
        np.savez(tmp, **kwargs)
        os.replace(tmp, out_npz)

    if os.path.exists(out_npz) and not args.fresh:
        try:
            z = np.load(out_npz, allow_pickle=False)
            prev_Ks = list(z['Ks'].tolist())
            prev_seeds = list(z['seeds'].tolist())
            if prev_Ks == Ks and prev_seeds == seeds:
                done = z['done'].astype(bool)
                panel_b_done = bool(z['panel_b_done'])
                for a in ALGOS:
                    stop_times[a] = z[f'{a}_stop']
                    correct[a] = z[f'{a}_correct'].astype(bool)
                    ct = z[f'{a}_curve_t']
                    cn = z[f'{a}_curve_n']
                    if ct.size > 0:
                        curves[a] = list(zip(ct.tolist(), cn.tolist()))
                hc = z['H_classical']
                hg = z['H_graph']
                H_classical = [None if np.isnan(v) else float(v) for v in hc]
                H_graph_vals = [None if np.isnan(v) else float(v) for v in hg]
                n_done = int(done.sum())
                total = len(Ks) * len(ALGOS)
                print(f"[resume] loaded {n_done}/{total} cells, "
                      f"panel_b_done={panel_b_done}", flush=True)
            else:
                print(f"[resume] checkpoint Ks={prev_Ks} seeds_n={len(prev_seeds)} "
                      f"does not match args; ignoring (use --fresh to silence)",
                      flush=True)
        except Exception as e:
            print(f"[resume] failed to load {out_npz}: {e}", flush=True)

    # ------------------------------------------------------------------
    # K-sweep
    # ------------------------------------------------------------------
    for ki, K in enumerate(Ks):
        mu, A, D = instances.union_of_cliques_with_challenger(K)
        if H_classical[ki] is None:
            H_classical[ki] = hardness.classical_hardness(mu)
            H_graph_vals[ki] = hardness.graph_hardness(mu, A, D, rho=1.0)
        print(f"\n=== K={K}: H_classical={H_classical[ki]:.2f}, "
              f"H_graph={H_graph_vals[ki]:.2f} ===", flush=True)

        for ai, name in enumerate(ALGOS):
            if done[ki, ai]:
                ts = stop_times[name][ki, :]
                cor = correct[name][ki, :]
                print(f"  {name:11s} [resumed] t_med={np.median(ts):8.0f}  "
                      f"correct={cor.mean():.0%}", flush=True)
                continue
            fac = make_factory(name, D, A, mu, delta=delta, q=args.q)
            t0 = time.time()
            runs = runners.run_many(fac, seeds, n_jobs=args.n_jobs,
                                    max_steps=args.max_steps,
                                    record_elimination=False,
                                    progress=False)
            ts = np.array([r['stopping_time'] for r in runs], dtype=float)
            cor = np.array([r['correct'] for r in runs], dtype=bool)
            stop_times[name][ki, :] = ts
            correct[name][ki, :] = cor
            done[ki, ai] = True
            save_checkpoint()
            elapsed = time.time() - t0
            print(f"  {name:11s} t_med={np.median(ts):8.0f}  "
                  f"IQR=[{np.percentile(ts,25):.0f}, {np.percentile(ts,75):.0f}]  "
                  f"correct={cor.mean():.0%}  ({elapsed:.0f}s)", flush=True)

    # ------------------------------------------------------------------
    # Single-seed elimination/agreement curves at K = panel_b_K
    # ------------------------------------------------------------------
    K_b = args.panel_b_K
    if panel_b_done:
        print(f"\n[panel B] [resumed] curves at K={K_b}, seed={args.panel_b_seed}",
              flush=True)
    else:
        print(f"\n[panel B] elimination curves at K={K_b}, "
              f"seed={args.panel_b_seed}", flush=True)
        mu_b, A_b, D_b = instances.union_of_cliques_with_challenger(K_b)
        for name in ALGOS:
            fac = make_factory(name, D_b, A_b, mu_b, delta=delta, q=args.q)
            out = runners.run_algorithm(fac, seed=args.panel_b_seed,
                                        max_steps=args.max_steps,
                                        record_elimination=True)
            curves[name] = out['elimination_curve']
            save_checkpoint()
            print(f"  {name}: t={out['stopping_time']}  "
                  f"curve_pts={len(curves[name])}", flush=True)
        panel_b_done = True
        save_checkpoint()

    print(f"\nSaved {out_npz}")
    print("Run experiments/main_1_plot.py to render the figure.")

    # Acceptance summary (numeric only; figure rendering happens in plot script).
    med = {a: np.median(stop_times[a], axis=1) for a in ALGOS}
    grub_ratio = med['GRUB'] / np.maximum(med['TS-Explore'], 1.0)
    print("\n# Acceptance")
    print(f"  TS-Explore median: {med['TS-Explore']}")
    print(f"  Basic TS median:   {med['Basic TS']}")
    print(f"  GRUB median:       {med['GRUB']}")
    print(f"  GRUB / TS-Explore ratios: {grub_ratio}")
    ok = bool(np.all(grub_ratio >= 5.0))
    print(f"  [{'x' if ok else ' '}] GRUB / TS-Explore >= 5x at every K  "
          f"(min ratio = {grub_ratio.min():.2f})")


if __name__ == "__main__":
    main()
