"""main_2 analytical pre-flight.

Reports H_classical, H_graph, smoothness eps, gap structure, and
predicted Basic-TS / TS-Explore ratios on the clustered_chain instance
across K = 20, 50, 100.  No bandit runs --- just closed-form hardness
and graph diagnostics.  Use to sanity-check the design before kicking
off the long server run.
"""
from __future__ import annotations

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
sys.path.insert(0, _REPO)

from experiments.utils import instances, hardness  # noqa: E402


def report_one(K, C=4, gap_step=0.3, rho=1.0, delta=1e-3):
    mu, A, D = instances.clustered_chain(K, C=C, gap_step=gap_step)
    a_star = int(np.argmax(mu))
    Delta = mu[a_star] - mu
    nz_gaps = sorted(set(np.round(Delta[Delta > 0], 6).tolist()))
    L = D - A
    eps2 = float(mu @ L @ mu)
    eps = float(np.sqrt(max(eps2, 0.0)))
    deg = np.diag(D)
    n_edges = int(A.sum() / 2)
    H_cls = hardness.classical_hardness(mu)
    H_gr = hardness.graph_hardness(mu, A, D, rho=rho)
    H_set, N_set = hardness.competitive_set(mu, A, D, rho=rho)

    cluster_sizes = []
    for g in nz_gaps:
        cluster_sizes.append(int(np.sum(np.isclose(Delta, g))))

    print(f"\n========= K = {K} (C={C}, gap_step={gap_step}) =========")
    print(f"  best arm at index {a_star}, mu = {mu[a_star]:.3f}")
    print(f"  cluster gaps  : {nz_gaps}")
    print(f"  cluster sizes : {cluster_sizes}")
    print(f"  smallest gap  : {min(nz_gaps):.3f}  -> 1/Delta^2 = "
          f"{1.0/min(nz_gaps)**2:.2f}")
    print(f"  edges = {n_edges},  deg(best)={int(deg[a_star])},  "
          f"max deg = {int(deg.max())}")
    print(f"  epsilon = sqrt(mu^T L mu)             = {eps:.3f}")
    print(f"  H_classical = sum 1/Delta^2           = {H_cls:.2f}")
    print(f"  H_graph     (rho={rho:.0f})              = {H_gr:.2f}")
    print(f"  H_classical / H_graph                 = {H_cls/H_gr:.2f}x")
    print(f"  competitive arms : {len(H_set)}/"
          f"{len(H_set)+len(N_set)} suboptimal")
    return dict(K=K, eps=eps, H_classical=H_cls, H_graph=H_gr,
                ratio=H_cls/H_gr)


def main():
    print("clustered_chain analytical pre-flight")
    print("=====================================")
    rows = []
    for C in [2, 3, 4]:
        print(f"\n\n##### C = {C} #####")
        for K in [20, 50, 100]:
            rows.append(report_one(K, C=C))

    print("\n========= summary =========")
    print(f"{'K':>5}  {'eps':>6}  {'H_cls':>10}  {'H_graph':>10}  "
          f"{'ratio':>8}")
    for r in rows:
        print(f"{r['K']:>5}  {r['eps']:>6.3f}  {r['H_classical']:>10.2f}  "
              f"{r['H_graph']:>10.2f}  {r['ratio']:>7.2f}x")
    print()
    print("Acceptance criteria for a graph-helpful instance:")
    print("  - epsilon small    (graph is smooth)")
    print("  - H_classical / H_graph grows with K  (graph helps more at large K)")
    print("  - smallest gap fixed across K (challenger gap is the bottleneck)")


if __name__ == "__main__":
    main()
