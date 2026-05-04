"""chembl_2 -- multi-target rho-sweep BAI experiment on ChEMBL.

Single-file runner.  For each of three ChEMBL targets, fetches/loads the
cached top-100 most-active-ligand instance, runs four BAI algorithms
across a small log-spaced rho grid, and saves seed-level stopping times
to ``experiments/outputs/chembl_2_results.npz``.  Plotting lives in
``chembl_2_plot.py``.

Algorithms (2x2 design: stopping rule x graph-aware estimator):
              | TS stopping        | UCB-LCB stopping
    --------- + -----------------  + ------------------
    no graph  | Basic TS           | KL-LUCB
    +graph    | TS-Explore         | GRUB
TS-Explore and GRUB depend on rho; Basic TS and KL-LUCB do not (they're
broadcast across rhos in the saved arrays for plotting convenience).

Targets (chosen for protein-family diversity AND clean gap structure;
candidates were screened in old/_screen_targets.py):
    CHEMBL204  -- thrombin, serine protease    (H_cls/H_g = 2.78x)
    CHEMBL325  -- histamine H3 receptor, GPCR  (H_cls/H_g = 1.94x)
    CHEMBL230  -- carbonic anhydrase II, zinc  (H_cls/H_g = 1.39x)
metalloenzyme.  Several otherwise-natural picks (CHEMBL244, CHEMBL279,
CHEMBL220) were rejected because their top-100 sets contain ties or
near-ties at the best arm, making BAI ill-posed.

Defaults (matching the canonical chembl_1 instance config):
    --rhos    1 3 10 30 100
    --seeds   20
    --top-k   100         (top-K most-active per target)
    --knn-k   10
    raw pIC50 (sigma=1 reward noise tracks the algorithm's assumed scale)

Usage:
    # one call runs the entire experiment (auto-fetches each target if not cached)
    python experiments/chembl_2.py

    # quick smoke
    python experiments/chembl_2.py --quick
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.utils import chembl_loader, hardness, runners  # noqa: E402
import graph_algo  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs')
os.makedirs(OUT, exist_ok=True)


DEFAULT_TARGETS = ['CHEMBL204', 'CHEMBL325', 'CHEMBL230']
DEFAULT_RHOS = [1.0, 3.0, 10.0, 30.0, 100.0]

# Algorithm classification.  RHO_DEPENDENT algorithms get one run per
# (target, rho); RHO_FREE algorithms run once per target and broadcast.
RHO_DEPENDENT = ['TS-Explore', 'GRUB']
RHO_FREE = ['Basic TS', 'KL-LUCB']
ALL_ALGOS = RHO_DEPENDENT + RHO_FREE


# ----------------------------------------------------------------------
# Picklable factories (multiprocessing requires top-level callables).
# ----------------------------------------------------------------------
class TSExploreFactory:
    def __init__(self, D, A, mu, delta, q, rho):
        self.kw = dict(D=D, A=A, mu=mu, rho_lap=rho, delta=delta, q=q)

    def __call__(self):
        return graph_algo.ThompsonSampling(**self.kw)


class GRUBFactory:
    def __init__(self, D, A, mu, delta, rho):
        self.kw = dict(D=D, A=A, mu=mu, rho_lap=rho, delta=delta)

    def __call__(self):
        return graph_algo.MaxVarianceArmAlgo(**self.kw)


class BasicTSFactory:
    def __init__(self, mu, delta, q):
        self.kw = dict(mu=mu, delta=delta, q=q)

    def __call__(self):
        return graph_algo.BasicThompsonSampling(**self.kw)


class _KLLUCBPaperSchedule(graph_algo.KL_LUCB):
    """KL-LUCB with the paper's L_1(t) = log(12 K^2 t^2 / delta)
    confidence schedule.

    The shared graph_algo.KL_LUCB uses the original
    Kaufmann-Kalyanakrishnan schedule log(K t^2 / delta), which is
    correct for that algorithm's published analysis but is tighter
    than the schedule used by BasicThompsonSampling and
    ThompsonSampling in this paper.  For the chembl_2 experiment we
    want all four algorithms to use the same confidence schedule so
    the comparison reflects only stopping-rule and graph-structure
    differences, not differences in confidence-bound analysis.

    This subclass is local to the chembl_2 experiment so other
    experiments (e.g. movielens) that import graph_algo.KL_LUCB are
    unaffected.
    """

    def _confidence_width(self):
        t_safe = max(float(self.t), 1.0)
        log_term = max(
            np.log(12.0 * (self.K ** 2) * (t_safe ** 2) / self.delta),
            1.0,
        )
        return self.sigma * np.sqrt(
            2.0 * log_term / np.maximum(self.counts, 1.0)
        )


class KLLUCBFactory:
    def __init__(self, mu, delta):
        self.kw = dict(mu=mu, delta=delta)

    def __call__(self):
        return _KLLUCBPaperSchedule(**self.kw)


def make_factory(name, D, A, mu, delta, q, rho):
    if name == 'TS-Explore':
        return TSExploreFactory(D, A, mu, delta, q, rho)
    if name == 'GRUB':
        return GRUBFactory(D, A, mu, delta=delta, rho=rho)
    if name == 'Basic TS':
        return BasicTSFactory(mu, delta, q)
    if name == 'KL-LUCB':
        return KLLUCBFactory(mu, delta)
    raise ValueError(name)


# ----------------------------------------------------------------------
# Cache management
# ----------------------------------------------------------------------
def cache_path_for(target):
    return os.path.join(OUT, f'chembl_{target.replace("CHEMBL", "")}_data.npz')


def load_or_build_target(target, top_k, knn_k, max_pages, verbose=True):
    """Return (mu, A, D) for a target, building+caching from the API if
    the cached .npz does not already exist."""
    path = cache_path_for(target)
    if os.path.exists(path):
        z = np.load(path, allow_pickle=False)
        if int(z['pIC50'].shape[0]) == top_k:
            if verbose:
                print(f"  [cache] using {path} (K={top_k})", flush=True)
            return z['pIC50'].astype(float), z['A'].astype(float), z['D'].astype(float)
        if verbose:
            print(f"  [cache] {path} has K={z['pIC50'].shape[0]} != "
                  f"{top_k}; rebuilding", flush=True)

    if verbose:
        print(f"  [cache] building {path} from ChEMBL API "
              f"(target={target}, K={top_k})", flush=True)
    payload = chembl_loader.build_instance(
        target=target, subsample_k=top_k, knn_k=knn_k,
        normalize=False, max_pages=max_pages, select='top',
        verbose=verbose)
    tmp = path + '.tmp.npz'
    np.savez(tmp, **payload)
    os.replace(tmp, path)
    return (payload['pIC50'].astype(float),
            payload['A'].astype(float),
            payload['D'].astype(float))


# ----------------------------------------------------------------------
# Main runner
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--targets', type=str, nargs='+',
                        default=DEFAULT_TARGETS)
    parser.add_argument('--rhos', type=float, nargs='+',
                        default=DEFAULT_RHOS)
    parser.add_argument('--seeds', type=int, default=20)
    parser.add_argument('--n-jobs', type=int, default=1)
    parser.add_argument('--q', type=float, default=0.1)
    parser.add_argument('--top-k', type=int, default=100,
                        help="number of molecules per target")
    parser.add_argument('--knn-k', type=int, default=10,
                        help="k-NN graph degree per target")
    parser.add_argument('--max-pages', type=int, default=5,
                        help="ChEMBL API pagination cap (5 pages = 5000 "
                             "records, ample for top-K=100)")
    parser.add_argument('--max-steps', type=int, default=5_000_000)
    parser.add_argument('--algos', type=str, nargs='+', default=ALL_ALGOS,
                        choices=ALL_ALGOS)
    parser.add_argument('--quick', action='store_true',
                        help="1 target, 2 rhos, 3 seeds, all algos")
    parser.add_argument('--fresh', action='store_true',
                        help="ignore any existing checkpoint and start over")
    parser.add_argument('--out', type=str,
                        default=os.path.join(OUT, 'chembl_2_results.npz'))
    args = parser.parse_args()

    if args.quick:
        targets = args.targets[:1]
        rhos = [1.0, 10.0]
        seeds = list(range(3))
        max_steps = min(args.max_steps, 1_000_000)
        algos = args.algos
    else:
        targets = list(args.targets)
        rhos = list(args.rhos)
        seeds = list(range(args.seeds))
        max_steps = args.max_steps
        algos = list(args.algos)

    delta = 1e-3
    n_targets = len(targets)
    n_rhos = len(rhos)
    n_seeds = len(seeds)
    out_npz = args.out

    print(f"[chembl_2] targets={targets}  rhos={rhos}  seeds={n_seeds}  "
          f"algos={algos}", flush=True)

    # ------------------------------------------------------------------
    # Per-target instance loading + per-target diagnostics
    # ------------------------------------------------------------------
    mus = {}
    Adjs = {}
    Degs = {}
    eps_L = np.full(n_targets, np.nan)
    H_classical_arr = np.full(n_targets, np.nan)
    H_graph_arr = np.full(n_targets, np.nan)
    smallest_gap_arr = np.full(n_targets, np.nan)
    for ti, target in enumerate(targets):
        print(f"\n=== target={target} ===", flush=True)
        mu, A, D = load_or_build_target(target, args.top_k, args.knn_k,
                                        args.max_pages, verbose=True)
        mus[target] = mu
        Adjs[target] = A
        Degs[target] = D
        L = D - A
        eps_L[ti] = float(np.sqrt(max(mu @ L @ mu, 0.0)))
        a_star = int(np.argmax(mu))
        Delta = mu[a_star] - mu
        nz = Delta[Delta > 0]
        smallest_gap_arr[ti] = float(nz.min()) if nz.size else np.nan
        H_classical_arr[ti] = float(hardness.classical_hardness(mu))
        H_graph_arr[ti] = float(hardness.graph_hardness(mu, A, D, rho=1.0))
        print(f"  K = {mu.shape[0]}  smallest_gap = "
              f"{smallest_gap_arr[ti]:.3f}  eps_L = {eps_L[ti]:.2f}  "
              f"H_cls = {H_classical_arr[ti]:.2f}  "
              f"H_graph(rho=1) = {H_graph_arr[ti]:.2f}", flush=True)

    # ------------------------------------------------------------------
    # State (with checkpoint resume)
    # ------------------------------------------------------------------
    # stop_times[(target, algo)] is (n_rhos, n_seeds); for rho-free algos
    # the rows are identical (broadcast at save time).
    stop_times = {(t, a): np.full((n_rhos, n_seeds), np.nan)
                  for t in targets for a in ALL_ALGOS}
    correct = {(t, a): np.zeros((n_rhos, n_seeds), dtype=bool)
               for t in targets for a in ALL_ALGOS}
    # done[(target, rho_idx, algo)] -- True if cell has been computed.
    # For rho-free algos, all rho_idx for a given target share one
    # computation, signalled by the "any True" in the column.
    done = {(t, a): np.zeros(n_rhos, dtype=bool)
            for t in targets for a in ALL_ALGOS}
    rho_free_done = {(t, a): False
                     for t in targets for a in RHO_FREE}

    INSTANCE_TAG = (f"top{args.top_k}_knn{args.knn_k}_K{args.top_k}_"
                    f"rhos{'_'.join(f'{r:g}' for r in rhos)}")

    def save_checkpoint():
        kwargs = dict(
            targets=np.asarray(targets, dtype='<U20'),
            rhos=np.array(rhos, dtype=float),
            seeds=np.array(seeds),
            top_k=int(args.top_k),
            knn_k=int(args.knn_k),
            delta=delta, q=args.q,
            instance=np.array(INSTANCE_TAG),
            eps_L=eps_L, H_classical=H_classical_arr,
            H_graph=H_graph_arr, smallest_gap=smallest_gap_arr,
        )
        for ti, t in enumerate(targets):
            for a in ALL_ALGOS:
                key = f'{t}__{a}'
                kwargs[f'{key}__stop'] = stop_times[(t, a)]
                kwargs[f'{key}__correct'] = correct[(t, a)].astype(int)
                kwargs[f'{key}__done'] = done[(t, a)].astype(int)
            kwargs[f'{t}__mu'] = mus[t]
        tmp = out_npz + '.tmp.npz'
        np.savez(tmp, **kwargs)
        os.replace(tmp, out_npz)

    if os.path.exists(out_npz) and not args.fresh:
        try:
            zr = np.load(out_npz, allow_pickle=False)
            prev_targets = list(zr['targets'].tolist())
            prev_rhos = list(zr['rhos'].tolist())
            prev_seeds = list(zr['seeds'].tolist())
            prev_inst = (str(zr['instance']) if 'instance' in zr.files
                         else '<unknown>')
            if (prev_targets == targets and prev_rhos == rhos
                    and prev_seeds == seeds
                    and prev_inst == INSTANCE_TAG):
                for t in targets:
                    for a in ALL_ALGOS:
                        key = f'{t}__{a}'
                        stop_times[(t, a)] = zr[f'{key}__stop']
                        correct[(t, a)] = zr[f'{key}__correct'].astype(bool)
                        done[(t, a)] = zr[f'{key}__done'].astype(bool)
                        if a in RHO_FREE:
                            rho_free_done[(t, a)] = bool(done[(t, a)].all())
                n_cells = sum(int(done[(t, a)].sum())
                              for t in targets for a in ALL_ALGOS)
                total = n_targets * n_rhos * len(ALL_ALGOS)
                print(f"\n[resume] loaded {n_cells}/{total} cells",
                      flush=True)
            else:
                print(f"\n[resume] checkpoint mismatch "
                      f"(prev_inst={prev_inst!r}); ignoring "
                      f"(use --fresh to silence)", flush=True)
        except Exception as e:
            print(f"\n[resume] failed to load {out_npz}: {e}", flush=True)

    # ------------------------------------------------------------------
    # Per-target sweep
    # ------------------------------------------------------------------
    for ti, target in enumerate(targets):
        mu = mus[target]
        A = Adjs[target]
        D = Degs[target]
        print(f"\n##### target={target}  K={mu.shape[0]} #####", flush=True)

        # Rho-free algorithms first (run once per target).
        for name in algos:
            if name not in RHO_FREE:
                continue
            if rho_free_done[(target, name)]:
                row = stop_times[(target, name)][0, :]
                cor = correct[(target, name)][0, :]
                print(f"  {name:11s} [resumed] t_med="
                      f"{np.median(row):8.0f}  "
                      f"correct={cor.mean():.0%}", flush=True)
                continue
            fac = make_factory(name, D, A, mu, delta=delta, q=args.q,
                               rho=1.0)  # rho ignored for these
            t0 = time.time()
            runs = runners.run_many(fac, seeds, n_jobs=args.n_jobs,
                                    max_steps=max_steps,
                                    record_elimination=False,
                                    progress=False)
            ts = np.array([r['stopping_time'] for r in runs], dtype=float)
            cor = np.array([r['correct'] for r in runs], dtype=bool)
            for ri in range(n_rhos):
                stop_times[(target, name)][ri, :] = ts
                correct[(target, name)][ri, :] = cor
                done[(target, name)][ri] = True
            rho_free_done[(target, name)] = True
            save_checkpoint()
            elapsed = time.time() - t0
            print(f"  {name:11s} t_med={np.median(ts):8.0f}  "
                  f"IQR=[{np.percentile(ts,25):.0f}, "
                  f"{np.percentile(ts,75):.0f}]  "
                  f"correct={cor.mean():.0%}  ({elapsed:.0f}s)",
                  flush=True)

        # Rho-dependent algorithms next.
        for ri, rho in enumerate(rhos):
            print(f"\n  --- rho = {rho:g} ---", flush=True)
            for name in algos:
                if name not in RHO_DEPENDENT:
                    continue
                if done[(target, name)][ri]:
                    row = stop_times[(target, name)][ri, :]
                    cor = correct[(target, name)][ri, :]
                    print(f"    {name:11s} [resumed] "
                          f"t_med={np.median(row):8.0f}  "
                          f"correct={cor.mean():.0%}", flush=True)
                    continue
                fac = make_factory(name, D, A, mu, delta=delta,
                                   q=args.q, rho=rho)
                t0 = time.time()
                runs = runners.run_many(fac, seeds, n_jobs=args.n_jobs,
                                        max_steps=max_steps,
                                        record_elimination=False,
                                        progress=False)
                ts = np.array([r['stopping_time'] for r in runs],
                              dtype=float)
                cor = np.array([r['correct'] for r in runs], dtype=bool)
                stop_times[(target, name)][ri, :] = ts
                correct[(target, name)][ri, :] = cor
                done[(target, name)][ri] = True
                save_checkpoint()
                elapsed = time.time() - t0
                print(f"    {name:11s} t_med={np.median(ts):8.0f}  "
                      f"IQR=[{np.percentile(ts,25):.0f}, "
                      f"{np.percentile(ts,75):.0f}]  "
                      f"correct={cor.mean():.0%}  ({elapsed:.0f}s)",
                      flush=True)

    # ------------------------------------------------------------------
    # Acceptance summary
    # ------------------------------------------------------------------
    print(f"\nSaved {out_npz}")
    print("\n# Acceptance (medians over seeds, per target / rho)")
    print(f"{'target':<11s}  {'rho':>6s}  " +
          '  '.join(f'{a:>11s}' for a in ALL_ALGOS) +
          '  ' + 'Basic/TS-Expl')
    for ti, target in enumerate(targets):
        for ri, rho in enumerate(rhos):
            meds = {a: float(np.nanmedian(stop_times[(target, a)][ri, :]))
                    for a in ALL_ALGOS}
            ratio = (meds['Basic TS'] / max(meds['TS-Explore'], 1.0))
            row = '  '.join(f'{meds[a]:>11.0f}' for a in ALL_ALGOS)
            print(f"  {target:<9s}  {rho:>6.1f}  {row}  {ratio:>11.2f}x")
    return 0


if __name__ == "__main__":
    sys.exit(main())
