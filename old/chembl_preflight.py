"""ChEMBL bandit instance pre-flight diagnostics.

Loads the cached instance ``.npz`` produced by
``experiments/utils/chembl_loader.py`` and reports:

  * graph diagnostics (nodes, edges, degree distribution, connected
    components, max degree)
  * smoothness diagnostics
        epsilon_L = sqrt(<mu, L_G mu>)         (combinatorial)
        epsilon_K = sqrt(<mu, K_G mu>)         (normalized)
  * gap structure: best, runner-up, smallest non-zero gap, gap quantiles
  * H_classical, H_graph (rho=1)
  * predicted optimal rho^*(epsilon_L) per the theorem

No bandit runs --- just closed-form summaries to confirm the instance
is a sensible target for the experiment before kicking off the long run.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
sys.path.insert(0, _REPO)

from experiments.utils import hardness  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str,
                        default=os.path.join(_REPO, 'experiments', 'outputs',
                                             'chembl_204_data.npz'))
    parser.add_argument('--rho', type=float, default=1.0)
    parser.add_argument('--delta', type=float, default=1e-3)
    parser.add_argument('--T-est', type=float, default=1e5,
                        help="estimate of stopping time T for L1(T) "
                             "computation in rho^*(eps)")
    args = parser.parse_args()

    if not os.path.exists(args.data):
        print(f"Error: {args.data} not found.  Run "
              f"experiments/utils/chembl_loader.py first.", file=sys.stderr)
        sys.exit(1)
    z = np.load(args.data, allow_pickle=False)
    target = str(z['target'])
    mu = z['pIC50'].astype(float)
    mu_raw = z['pIC50_raw'].astype(float)
    A = z['A'].astype(float)
    D = z['D'].astype(float)
    chembl_ids = z['chembl_ids']
    K = mu.shape[0]
    knn_k = int(z['knn_k'])

    deg = np.diag(D)
    n_edges = int(A.sum() / 2)

    a_star = int(np.argmax(mu))
    Delta = mu[a_star] - mu
    nz_gaps = np.sort(Delta[Delta > 0])
    smallest_gap = float(nz_gaps[0]) if nz_gaps.size > 0 else float('nan')
    gap_q25 = float(np.percentile(nz_gaps, 25)) if nz_gaps.size else float('nan')
    gap_q50 = float(np.percentile(nz_gaps, 50)) if nz_gaps.size else float('nan')

    L = D - A
    eps_L = float(np.sqrt(max(mu @ L @ mu, 0.0)))
    d_inv_sqrt = np.where(deg > 0, 1.0 / np.sqrt(np.maximum(deg, 1e-12)), 0.0)
    K_norm = np.eye(K) - (d_inv_sqrt[:, None] * A) * d_inv_sqrt[None, :]
    eps_K = float(np.sqrt(max(mu @ K_norm @ mu, 0.0)))

    H_cls = hardness.classical_hardness(mu)
    H_gr = hardness.graph_hardness(mu, A, D, rho=args.rho)
    H_set, N_set = hardness.competitive_set(mu, A, D, rho=args.rho)

    rho_star_L = hardness.rho_star(eps_L, K, args.T_est, args.delta)
    rho_star_K = hardness.rho_star(eps_K, K, args.T_est, args.delta)

    # Connected components.
    import networkx as nx
    G = nx.from_numpy_array(A)
    n_cc = nx.number_connected_components(G)
    largest_cc = max((len(c) for c in nx.connected_components(G)), default=0)

    print(f"========= ChEMBL pre-flight  ({target}) =========")
    print(f"  source file       : {args.data}")
    print(f"  K  (arms)         : {K}")
    print(f"  edges             : {n_edges}    "
          f"(k-NN k = {knn_k}, mean deg = {deg.mean():.1f}, "
          f"max deg = {int(deg.max())})")
    print(f"  components        : {n_cc} "
          f"(largest = {largest_cc} / {K})")
    print()
    print(f"  pIC50 (raw)       : range [{mu_raw.min():.2f}, "
          f"{mu_raw.max():.2f}], median {np.median(mu_raw):.2f}")
    print(f"  pIC50 (normalized): range [{mu.min():.3f}, {mu.max():.3f}]"
          f", median {np.median(mu):.3f}")
    print()
    print(f"  best arm idx      : {a_star} "
          f"(ChEMBL id = {chembl_ids[a_star]})")
    print(f"  best mu (norm)    : {mu[a_star]:.4f}")
    print(f"  smallest gap      : {smallest_gap:.4f}  "
          f"-> 1/Delta^2 = {1.0 / max(smallest_gap, 1e-9)**2:.2f}")
    print(f"  25/50%ile gap     : {gap_q25:.4f} / {gap_q50:.4f}")
    print(f"  best arm degree   : {int(deg[a_star])}")
    print()
    print(f"  epsilon_L (combinatorial)   = {eps_L:.3f}")
    print(f"  epsilon_K (normalized)      = {eps_K:.3f}")
    print(f"  H_classical                 = {H_cls:.2f}")
    print(f"  H_graph (rho={args.rho:.0f})           = {H_gr:.2f}")
    print(f"  H_classical / H_graph       = "
          f"{H_cls / max(H_gr, 1e-9):.2f}x")
    print(f"  competitive arms            : {len(H_set)}/{K-1}")
    print()
    print(f"  predicted rho^*(eps_L)      = {rho_star_L:.1f}   "
          f"(T-est = {args.T_est:.0e})")
    print(f"  predicted rho^*(eps_K)      = {rho_star_K:.1f}")
    print()
    print(f"Acceptance heuristics:")
    print(f"  - smallest gap > 0                      "
          f"[{'x' if smallest_gap > 0 else ' '}]")
    print(f"  - graph connected (largest_cc == K)     "
          f"[{'x' if largest_cc == K else ' '}]  ({largest_cc}/{K})")
    print(f"  - smoothness eps_L < |mu|_2             "
          f"[{'x' if eps_L < float(np.linalg.norm(mu)) else ' '}]  "
          f"(eps_L = {eps_L:.3f}, ||mu|| = {np.linalg.norm(mu):.3f})")


if __name__ == "__main__":
    main()
