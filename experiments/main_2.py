"""main_2 -- graph-helpful K-sweep on the clustered_chain instance.

Companion to ``main_1.py``.  Where ``main_1`` validates the log^2(K
H_graph) scaling on a single-challenger instance --- which incidentally
makes Basic TS empirically tied with TS-Explore --- this experiment
targets the graph-regularized estimator's contribution by running on a
*chain of equal-mean cliques* (default C=2: best singleton bridged to
a single challenger clique).  All challenger-clique nodes share the
same true mean, so the Laplacian regularizer can pool their
observations and TS-Explore distinguishes the cluster from the best
arm using roughly cluster-aggregate samples while Basic TS pays for
each challenger individually.

Default rho = 100 is near the theorem-prescribed rho^*(eps) for
epsilon = gap_step on this instance; rho = 1 puts the estimator in a
data-dominated regime where the regularizer has negligible effect, so
the canonical rho-versus-Basic comparison must be made at a non-trivial
rho.  Same K-sweep [20, 50, 100] and same three algorithms as main_1
(TS-Explore, Basic TS, GRUB) for direct comparability of figures.

Saves all raw results to ``experiments/outputs/main_2_results.npz``.
Plotting lives in ``main_2_plot.py``.
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
    def __init__(self, D, A, mu, delta, q, rho):
        self.kw = dict(D=D, A=A, mu=mu, rho_lap=rho, delta=delta, q=q)

    def __call__(self):
        return graph_algo.ThompsonSampling(**self.kw)


class BasicTSFactory:
    def __init__(self, mu, delta, q):
        self.kw = dict(mu=mu, delta=delta, q=q)

    def __call__(self):
        return graph_algo.BasicThompsonSampling(**self.kw)


class GRUBFactory:
    def __init__(self, D, A, mu, delta, rho):
        self.kw = dict(D=D, A=A, mu=mu, rho_lap=rho, delta=delta)

    def __call__(self):
        return graph_algo.MaxVarianceArmAlgo(**self.kw)


ALGOS = ['TS-Explore', 'Basic TS', 'GRUB']
INSTANCE_TAG = 'clustered_chain'


def make_factory(name, D, A, mu, delta, q, rho):
    if name == 'TS-Explore':
        return TSExploreFactory(D, A, mu, delta, q, rho)
    if name == 'Basic TS':
        return BasicTSFactory(mu, delta, q)
    if name == 'GRUB':
        return GRUBFactory(D, A, mu, delta=delta, rho=rho)
    raise ValueError(name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--Ks', type=int, nargs='+',
                        default=[20, 50, 100],
                        help="K values to sweep (default: 20 50 100)")
    parser.add_argument('--seeds', type=int, default=20)
    parser.add_argument('--n-jobs', type=int, default=1)
    parser.add_argument('--q', type=float, default=0.1)
    parser.add_argument('--C', type=int, default=2,
                        help="number of clusters in clustered_chain "
                             "(includes the best-arm singleton as cluster 0); "
                             "default C=2 = best singleton plus a single "
                             "challenger clique")
    parser.add_argument('--gap-step', type=float, default=0.3,
                        help="per-cluster gap increment (cluster c at "
                             "mu = mu_best - c*gap_step)")
    parser.add_argument('--rho', type=float, default=100.0,
                        help="Laplacian regularization weight for "
                             "TS-Explore and GRUB; default rho=100 is near "
                             "the theorem-prescribed rho^*(eps) for "
                             "epsilon = gap_step on this instance")
    parser.add_argument('--max-steps', type=int, default=10_000_000,
                        help="hard cap on steps per run; Basic TS may need "
                             "this to be large on the challenger-cluster "
                             "instance")
    parser.add_argument('--quick', action='store_true',
                        help="K=[20], seeds=3, max-steps=2_000_000")
    parser.add_argument('--fresh', action='store_true',
                        help="ignore any existing checkpoint and start over")
    args = parser.parse_args()

    if args.quick:
        Ks = [20]
        seeds = list(range(3))
        args.max_steps = min(args.max_steps, 2_000_000)
    else:
        Ks = args.Ks
        seeds = list(range(args.seeds))

    delta = 1e-3
    out_npz = os.path.join(OUT, 'main_2_results.npz')

    # ------------------------------------------------------------------
    # State (with checkpoint resume)
    # ------------------------------------------------------------------
    stop_times = {a: np.full((len(Ks), len(seeds)), np.nan) for a in ALGOS}
    correct = {a: np.zeros((len(Ks), len(seeds)), dtype=bool) for a in ALGOS}
    H_classical = [None] * len(Ks)
    H_graph_vals = [None] * len(Ks)
    eps_vals = [None] * len(Ks)
    done = np.zeros((len(Ks), len(ALGOS)), dtype=bool)
    rho = float(args.rho)

    def save_checkpoint():
        kwargs = dict(
            Ks=np.array(Ks),
            seeds=np.array(seeds),
            H_classical=np.array(
                [v if v is not None else np.nan for v in H_classical]),
            H_graph=np.array(
                [v if v is not None else np.nan for v in H_graph_vals]),
            eps=np.array(
                [v if v is not None else np.nan for v in eps_vals]),
            instance=np.array(INSTANCE_TAG),
            C=int(args.C),
            gap_step=float(args.gap_step),
            rho=rho,
            delta=delta,
            q=args.q,
            done=done.astype(int),
        )
        for a in ALGOS:
            kwargs[f'{a}_stop'] = stop_times[a]
            kwargs[f'{a}_correct'] = correct[a].astype(int)
        tmp = out_npz + '.tmp.npz'
        np.savez(tmp, **kwargs)
        os.replace(tmp, out_npz)

    if os.path.exists(out_npz) and not args.fresh:
        try:
            z = np.load(out_npz, allow_pickle=False)
            prev_Ks = list(z['Ks'].tolist())
            prev_seeds = list(z['seeds'].tolist())
            prev_instance = (str(z['instance'])
                             if 'instance' in z.files else '<unknown>')
            prev_C = int(z['C']) if 'C' in z.files else -1
            prev_gap = (float(z['gap_step'])
                        if 'gap_step' in z.files else float('nan'))
            prev_rho = (float(z['rho'])
                        if 'rho' in z.files else float('nan'))
            if (prev_Ks == Ks and prev_seeds == seeds
                    and prev_instance == INSTANCE_TAG
                    and prev_C == int(args.C)
                    and abs(prev_gap - float(args.gap_step)) < 1e-9
                    and abs(prev_rho - rho) < 1e-9):
                done = z['done'].astype(bool)
                for a in ALGOS:
                    stop_times[a] = z[f'{a}_stop']
                    correct[a] = z[f'{a}_correct'].astype(bool)
                hc = z['H_classical']
                hg = z['H_graph']
                ev = z['eps']
                H_classical = [None if np.isnan(v) else float(v) for v in hc]
                H_graph_vals = [None if np.isnan(v) else float(v) for v in hg]
                eps_vals = [None if np.isnan(v) else float(v) for v in ev]
                n_done = int(done.sum())
                total = len(Ks) * len(ALGOS)
                print(f"[resume] loaded {n_done}/{total} cells", flush=True)
            else:
                print(f"[resume] checkpoint mismatch "
                      f"(prev_instance={prev_instance!r}, prev_C={prev_C}, "
                      f"prev_gap={prev_gap}, prev_rho={prev_rho}); ignoring "
                      f"(use --fresh to silence)", flush=True)
        except Exception as e:
            print(f"[resume] failed to load {out_npz}: {e}", flush=True)

    # ------------------------------------------------------------------
    # K-sweep
    # ------------------------------------------------------------------
    for ki, K in enumerate(Ks):
        mu, A, D = instances.clustered_chain(
            K, C=args.C, gap_step=args.gap_step)
        if H_classical[ki] is None:
            H_classical[ki] = hardness.classical_hardness(mu)
            H_graph_vals[ki] = hardness.graph_hardness(mu, A, D, rho=rho)
            L = D - A
            eps_vals[ki] = float(np.sqrt(max(mu @ L @ mu, 0.0)))
        print(f"\n=== K={K}: H_classical={H_classical[ki]:.2f}, "
              f"H_graph={H_graph_vals[ki]:.2f}, "
              f"eps={eps_vals[ki]:.3f} ===", flush=True)

        for ai, name in enumerate(ALGOS):
            if done[ki, ai]:
                ts = stop_times[name][ki, :]
                cor = correct[name][ki, :]
                print(f"  {name:11s} [resumed] t_med={np.median(ts):8.0f}  "
                      f"correct={cor.mean():.0%}", flush=True)
                continue
            fac = make_factory(name, D, A, mu, delta=delta, q=args.q,
                               rho=rho)
            t0 = time.time()
            runs = runners.run_many(fac, seeds, n_jobs=args.n_jobs,
                                    max_steps=args.max_steps,
                                    record_elimination=False,
                                    progress=False)
            ts = np.array([r['stopping_time'] for r in runs], dtype=float)
            cor = np.array([r['correct'] for r in runs], dtype=bool)
            converged = np.array([r['converged_flag'] for r in runs],
                                 dtype=bool)
            stop_times[name][ki, :] = ts
            correct[name][ki, :] = cor
            done[ki, ai] = True
            save_checkpoint()
            elapsed = time.time() - t0
            n_unconv = int((~converged).sum())
            unconv_str = (f", unconverged={n_unconv}/{len(seeds)}"
                          if n_unconv else "")
            print(f"  {name:11s} t_med={np.median(ts):8.0f}  "
                  f"IQR=[{np.percentile(ts,25):.0f}, "
                  f"{np.percentile(ts,75):.0f}]  "
                  f"correct={cor.mean():.0%}{unconv_str}  "
                  f"({elapsed:.0f}s)", flush=True)

    print(f"\nSaved {out_npz}")
    print("Run experiments/main_2_plot.py to render the figure.")

    # Acceptance summary.
    med = {a: np.median(stop_times[a], axis=1) for a in ALGOS}
    basic_ratio = med['Basic TS'] / np.maximum(med['TS-Explore'], 1.0)
    grub_ratio = med['GRUB'] / np.maximum(med['TS-Explore'], 1.0)
    print("\n# Acceptance")
    print(f"  TS-Explore median: {med['TS-Explore']}")
    print(f"  Basic TS median:   {med['Basic TS']}")
    print(f"  GRUB median:       {med['GRUB']}")
    print(f"  Basic TS / TS-Explore ratios: {basic_ratio}")
    print(f"  GRUB / TS-Explore ratios:     {grub_ratio}")
    growing = np.all(np.diff(basic_ratio) >= 0)
    print(f"  [{'x' if growing else ' '}] Basic TS / TS-Explore is "
          f"non-decreasing in K  "
          f"(values = {basic_ratio.tolist()})")


if __name__ == "__main__":
    main()
