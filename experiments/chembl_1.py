"""chembl_1 -- real-graph hero experiment on a ChEMBL bioactivity instance.

Loads the cached instance produced by
``experiments/utils/chembl_loader.py`` and runs TS-Explore, Basic TS,
and GRUB at a single specified rho on the same instance, across
multiple seeds, with checkpointed results.

On a real chemistry-like instance where pIC50 is approximately smooth
on the molecular-similarity graph, the graph-regularized TS-Explore
identifies the most-active compound with substantially fewer reward
queries than Basic TS.  GRUB serves as the same-estimator elimination
baseline.

Canonical end-to-end pipeline:
    # 1. Build the cached instance (top-100 most-active CHEMBL204
    #    ligands, 10-NN Tanimoto graph, raw pIC50).
    python -m experiments.utils.chembl_loader \
        --target CHEMBL204 --out experiments/outputs/chembl_204_data.npz

    # 2. Run the bandit experiment.
    python experiments/chembl_1.py --rho 3 --seeds 20

    # 3. Render the figure.
    python experiments/chembl_1_plot.py

Smoke-test acceptance numbers (3 seeds, K=100, rho=3, raw pIC50):
    TS-Explore   t_med =     6190
    Basic TS     t_med =    17241   ->  2.79x slower than TS-Explore
    GRUB         t_med =   152455   ->  24.63x slower
which matches the analytical H_classical / H_graph(rho=1) = 2.78x.

Saves raw results to ``experiments/outputs/chembl_1_results.npz``;
plotting lives in ``chembl_1_plot.py``.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.utils import hardness, runners  # noqa: E402
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
    parser.add_argument('--data', type=str,
                        default=os.path.join(OUT, 'chembl_204_data.npz'),
                        help="path to cached ChEMBL instance .npz")
    parser.add_argument('--seeds', type=int, default=20)
    parser.add_argument('--n-jobs', type=int, default=1)
    parser.add_argument('--q', type=float, default=0.1)
    parser.add_argument('--rho', type=float, default=3.0,
                        help="Laplacian regularization weight; default "
                             "rho=3 is the empirical sweet spot for the "
                             "canonical CHEMBL204 top-K=200 raw-pIC50 "
                             "instance (theorem-prescribed rho^*(eps_L) "
                             "is approximately 2.6 for that instance)")
    parser.add_argument('--max-steps', type=int, default=5_000_000)
    parser.add_argument('--algos', type=str, nargs='+', default=ALGOS,
                        choices=ALGOS)
    parser.add_argument('--quick', action='store_true',
                        help="seeds=3, max-steps=1_000_000")
    parser.add_argument('--fresh', action='store_true',
                        help="ignore any existing checkpoint and start over")
    args = parser.parse_args()

    if args.quick:
        seeds = list(range(3))
        args.max_steps = min(args.max_steps, 1_000_000)
    else:
        seeds = list(range(args.seeds))
    rho = float(args.rho)

    if not os.path.exists(args.data):
        print(f"Error: {args.data} not found.  Run "
              f"experiments/utils/chembl_loader.py first.", file=sys.stderr)
        sys.exit(1)

    z = np.load(args.data, allow_pickle=False)
    target = str(z['target'])
    mu = z['pIC50'].astype(float)
    A = z['A'].astype(float)
    D = z['D'].astype(float)
    K = mu.shape[0]
    delta = 1e-3

    # Pre-flight summary so the log captures the instance we ran on.
    L = D - A
    eps_L = float(np.sqrt(max(mu @ L @ mu, 0.0)))
    a_star = int(np.argmax(mu))
    Delta = mu[a_star] - mu
    nz_gaps = Delta[Delta > 0]
    smallest_gap = float(nz_gaps.min()) if nz_gaps.size else float('nan')
    H_cls = hardness.classical_hardness(mu)
    H_gr = hardness.graph_hardness(mu, A, D, rho=rho)
    print(f"[chembl_1] target={target}  K={K}  rho={rho}", flush=True)
    print(f"  smallest gap = {smallest_gap:.4f}, "
          f"epsilon_L = {eps_L:.3f}, "
          f"H_classical = {H_cls:.1f}, H_graph = {H_gr:.1f}", flush=True)

    out_npz = os.path.join(OUT, 'chembl_1_results.npz')
    INSTANCE_TAG = f"{target}_K{K}_rho{rho}"

    # ------------------------------------------------------------------
    # State (with checkpoint resume)
    # ------------------------------------------------------------------
    stop_times = {a: np.full(len(seeds), np.nan) for a in ALGOS}
    correct = {a: np.zeros(len(seeds), dtype=bool) for a in ALGOS}
    pull_counts = {a: np.zeros((len(seeds), K), dtype=int) for a in ALGOS}
    done = {a: False for a in ALGOS}

    def save_checkpoint():
        kwargs = dict(
            seeds=np.array(seeds),
            mu=mu, A=A, D=D,
            target=np.array(target, dtype='<U20'),
            instance=np.array(INSTANCE_TAG),
            K=int(K),
            rho=float(rho),
            delta=delta,
            q=args.q,
            eps_L=eps_L,
            H_classical=H_cls,
            H_graph=H_gr,
            smallest_gap=smallest_gap,
            best_arm=int(a_star),
        )
        for a in ALGOS:
            kwargs[f'{a}_stop'] = stop_times[a]
            kwargs[f'{a}_correct'] = correct[a].astype(int)
            kwargs[f'{a}_pulls'] = pull_counts[a]
            kwargs[f'{a}_done'] = int(done[a])
        tmp = out_npz + '.tmp.npz'
        np.savez(tmp, **kwargs)
        os.replace(tmp, out_npz)

    if os.path.exists(out_npz) and not args.fresh:
        try:
            zr = np.load(out_npz, allow_pickle=False)
            prev_seeds = list(zr['seeds'].tolist())
            prev_instance = (str(zr['instance'])
                             if 'instance' in zr.files else '<unknown>')
            if prev_seeds == seeds and prev_instance == INSTANCE_TAG:
                for a in ALGOS:
                    stop_times[a] = zr[f'{a}_stop']
                    correct[a] = zr[f'{a}_correct'].astype(bool)
                    pull_counts[a] = zr[f'{a}_pulls']
                    done[a] = bool(int(zr[f'{a}_done']))
                n_done = sum(done.values())
                print(f"[resume] loaded {n_done}/{len(ALGOS)} algos",
                      flush=True)
            else:
                print(f"[resume] checkpoint mismatch "
                      f"(prev_instance={prev_instance!r}, want "
                      f"{INSTANCE_TAG!r}); ignoring "
                      f"(use --fresh to silence)", flush=True)
        except Exception as e:
            print(f"[resume] failed to load {out_npz}: {e}", flush=True)

    # ------------------------------------------------------------------
    # Algorithm loop (no K-sweep; one instance, one rho).
    # ------------------------------------------------------------------
    for name in args.algos:
        if done[name]:
            ts = stop_times[name]
            cor = correct[name]
            print(f"  {name:11s} [resumed] t_med={np.nanmedian(ts):.0f}  "
                  f"correct={cor.mean():.0%}", flush=True)
            continue
        fac = make_factory(name, D, A, mu, delta=delta, q=args.q, rho=rho)
        t0 = time.time()
        runs = runners.run_many(fac, seeds, n_jobs=args.n_jobs,
                                max_steps=args.max_steps,
                                record_elimination=False, progress=False)
        ts = np.array([r['stopping_time'] for r in runs], dtype=float)
        cor = np.array([r['correct'] for r in runs], dtype=bool)
        converged = np.array([r['converged_flag'] for r in runs],
                             dtype=bool)
        for si, r in enumerate(runs):
            stop_times[name][si] = ts[si]
            correct[name][si] = cor[si]
            pulls = r.get('pull_counts')
            if pulls is not None and pulls.size == K:
                pull_counts[name][si, :] = pulls.astype(int)
        done[name] = True
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
    print("Run experiments/chembl_1_plot.py to render the figure.")

    # Acceptance summary.
    med = {a: float(np.nanmedian(stop_times[a])) for a in ALGOS}
    cor_rate = {a: float(np.mean(correct[a])) for a in ALGOS}
    print("\n# Acceptance")
    for a in ALGOS:
        print(f"  {a:11s}  t_med={med[a]:>10.0f}  correct={cor_rate[a]:.0%}")
    if med['TS-Explore'] > 0:
        print(f"  Basic TS / TS-Explore = "
              f"{med['Basic TS'] / med['TS-Explore']:.2f}x")
        print(f"  GRUB     / TS-Explore = "
              f"{med['GRUB'] / med['TS-Explore']:.2f}x")


if __name__ == "__main__":
    main()
