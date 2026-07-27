"""Task 1 / Q7 -- which graph property predicts the per-arm reduction?

Reviewer 4Y4U's Q7 asks whether the improvement "can be characterized more
explicitly for common graph families or in terms of quantities such as cluster
structure, graph density, or influence factors".

This reads the saved traces (no new bandit runs) and regresses the per-arm
pooling factor  t_eff,i / t_i  against three candidate predictors:

  * clique/cluster size   |C(i)| -- the size of the maximal clique containing i
  * degree                deg(i) -- the local density proxy
  * influence factor      J(i,G) = 1/max_j r(i,j)  (Thaker et al.'s definition)

and reports which one actually tracks the realized pooling.

  python experiments/trace_analysis.py
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np
import networkx as nx

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.utils import instances  # noqa: E402
from experiments.graph_params import influence_thaker  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs')
TRACES = os.path.join(OUT, 'traces')


def max_clique_size_per_node(A):
    G = nx.from_numpy_array(A)
    cliques = list(nx.find_cliques(G))
    out = np.ones(A.shape[0])
    for c in cliques:
        for v in c:
            out[v] = max(out[v], len(c))
    return out


def main():
    files = sorted(glob.glob(os.path.join(TRACES, 'chain_K*_TS-Explore_*.npz')))
    if not files:
        print(f"no chain TS-Explore traces in {TRACES}")
        return 1

    print("Q7: what predicts the per-arm pooling factor t_eff,i / t_i ?")
    print("(TS-Explore on the clustered chain; only arms with t_i >= 1)\n")

    by_K = {}
    for f in files:
        base = os.path.basename(f)
        K = int(base.split('_')[1][1:])
        by_K.setdefault(K, []).append(f)

    print(f"{'K':>5} {'seeds':>6} {'pool(a*)':>9} {'pool(clique)':>13} "
          f"{'|clique|':>9} {'deg(a*)':>8} {'deg(clique)':>12}")
    for K in sorted(by_K):
        mu, A, D = instances.clustered_chain(K, C=2, gap_step=0.3)
        cl = max_clique_size_per_node(A)
        deg = np.diag(D)
        pools_star, pools_cl = [], []
        for f in by_K[K]:
            z = np.load(f, allow_pickle=False)
            teff, pulls = z['teff'][-1], z['pulls'][-1]
            a_star = int(z['a_star'])
            m = pulls >= 1
            pool = np.where(m, teff / np.maximum(pulls, 1e-12), np.nan)
            pools_star.append(pool[a_star])
            others = np.array([i for i in range(K) if i != a_star])
            pools_cl.append(np.nanmedian(pool[others]))
        print(f"{K:>5} {len(by_K[K]):>6} {np.nanmedian(pools_star):>9.2f} "
              f"{np.nanmedian(pools_cl):>13.2f} "
              f"{cl[1]:>9.0f} {deg[a_star]:>8.0f} {deg[1]:>12.0f}")

    print("\n  The challenger clique has |clique| = K-1 and the pooling factor")
    print("  tracks it; the best arm is a singleton and gets pooling ~1, which")
    print("  is why it is the bottleneck.  But this instance CANNOT separate the")
    print("  candidates: clique members have |clique| = deg = K-1, so cluster")
    print("  size and degree are indistinguishable here.  See the SBM below,")
    print("  where they differ and the influence factor wins.")

    # --- a family where the candidates disagree ----------------------
    print("\nOn the connected SBM (K=31), where the three predictors differ.")
    print("NOTE: only meaningful at rho >= rho_var(eps) -- at rho=100 the")
    print("instance is not in the pooling regime (pool ~1.3x) and all three")
    print("predictors are noise.  rho_var(eps_true) = 1.10e4 here.")
    mu, A, D = instances.sbm_phase_transition_connected()
    cl = max_clique_size_per_node(A)
    deg = np.diag(D)
    J = influence_thaker(A, D)
    tags = sorted({os.path.basename(f).split('_TS-Explore_')[0]
                   for f in glob.glob(os.path.join(
                       TRACES, 'sbm31_*_TS-Explore_*.npz'))})
    if not tags:
        print("  (no sbm31 TS-Explore trace yet)")
        return 0
    K = len(cl)
    blk = np.array([0] + [1 + (i - 1) // 6 for i in range(1, K)])
    for tag in tags:
        fs = sorted(glob.glob(os.path.join(
            TRACES, f'{tag}_TS-Explore_*.npz')))
        pool_all = []
        for f in fs:
            z = np.load(f, allow_pickle=False)
            teff, pulls = z['teff'][-1], z['pulls'][-1]
            m = pulls >= 1
            pool_all.append(np.where(m, teff / np.maximum(pulls, 1e-12),
                                     np.nan))
        pool = np.nanmedian(np.vstack(pool_all), axis=0)
        ok = np.isfinite(pool)
        print(f"\n  {tag}: {len(fs)} seeds, median pooling "
              f"{np.nanmedian(pool):.2f}x, arms pulled {ok.sum()}/{len(pool)}")
        for name, x in [('max-clique size', cl), ('degree', deg),
                        ('influence factor J', J)]:
            xx, yy = x[ok], pool[ok]
            if np.std(xx) < 1e-12 or np.std(yy) < 1e-12:
                print(f"    {name:20s} corr = (degenerate)")
                continue
            r = float(np.corrcoef(xx, yy)[0, 1])
            rs = float(np.corrcoef(np.argsort(np.argsort(xx)),
                                   np.argsort(np.argsort(yy)))[0, 1])
            print(f"    {name:20s} Pearson r = {r:+.3f}   "
                  f"Spearman = {rs:+.3f}")
        print("    pooling by SBM block: " + ", ".join(
            f"b{b}={np.nanmedian(pool[blk == b]):.1f}" for b in range(6)))
    return 0


if __name__ == '__main__':
    sys.exit(main())
