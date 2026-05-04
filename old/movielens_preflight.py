"""MovieLens-100K analytical pre-flight.

Reports per-K diagnostics for the real-world bandit instance:
    - epsilon = sqrt(<mu, L_G mu>)         (smoothness on graph)
    - smallest gap and 1/Delta_min^2
    - H_classical, H_graph at rho=1
    - degree statistics
    - rating-count statistics (sanity: all top-K items have many ratings)
    - top-rated movies (qualitative check)

No bandit runs --- closed-form hardness only.  Use to confirm the
instance has a non-trivial graph and a defensible BAI structure before
kicking off the experiment.
"""
from __future__ import annotations

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
sys.path.insert(0, _REPO)

from experiments.utils import movielens, hardness  # noqa: E402


def report_one(K, top_k_neighbors=5, min_common=5):
    mu, A, D, meta = movielens.build_instance(
        K=K, top_k_neighbors=top_k_neighbors, min_common=min_common,
        return_meta=True,
    )
    a_star = int(np.argmax(mu))
    Delta = mu[a_star] - mu
    Delta_pos = Delta[Delta > 0]
    deg = np.diag(D)
    L = D - A
    eps2 = float(mu @ L @ mu)
    eps = float(np.sqrt(max(eps2, 0.0)))
    n_edges = int(A.sum() / 2)
    H_cls = hardness.classical_hardness(mu)
    H_gr = hardness.graph_hardness(mu, A, D, rho=1.0)
    H_set, N_set = hardness.competitive_set(mu, A, D, rho=1.0)
    n_isolated = int((deg == 0).sum())

    print(f"\n========= K = {K} (top_k_neighbors={top_k_neighbors}) =========")
    print(f"  best arm idx={a_star}  mu_best={mu[a_star]:.3f}")
    if a_star < len(meta['titles']):
        print(f"    title    : {meta['titles'][a_star]}")
        print(f"    raters   : {int(meta['rating_counts'][a_star])}")
    print(f"  smallest gap   : {Delta_pos.min():.4f}  -> "
          f"1/Delta^2 = {1.0/Delta_pos.min()**2:.1f}")
    print(f"  median gap     : {np.median(Delta_pos):.3f}")
    print(f"  largest gap    : {Delta_pos.max():.3f}")
    print(f"  edges          : {n_edges}  "
          f"(avg deg = {2*n_edges/K:.1f}, max deg = {int(deg.max())}, "
          f"isolated nodes = {n_isolated})")
    print(f"  rating counts  : min={int(meta['rating_counts'].min())}, "
          f"med={int(np.median(meta['rating_counts']))}, "
          f"max={int(meta['rating_counts'].max())}")
    print(f"  epsilon        : {eps:.3f}  (epsilon^2 = {eps2:.3f})")
    print(f"  H_classical    : {H_cls:.2f}")
    print(f"  H_graph (rho=1): {H_gr:.2f}")
    print(f"  H_cls / H_graph: {H_cls/max(H_gr,1e-9):.2f}x")
    print(f"  competitive    : {len(H_set)}/{len(H_set)+len(N_set)} suboptimal")
    # Top 5 movies
    top5 = np.argsort(mu)[::-1][:5]
    print(f"  top 5 by mu_i:")
    for j in top5:
        title = meta['titles'][j] if j < len(meta['titles']) else f"#{j}"
        print(f"    mu={mu[j]:.3f}  raters={int(meta['rating_counts'][j])}  "
              f"deg={int(deg[j])}  -- {title}")
    return dict(K=K, eps=eps, H_classical=H_cls, H_graph=H_gr,
                gap_min=float(Delta_pos.min()),
                n_edges=n_edges, n_isolated=n_isolated)


def report_rho_sweep(K, rhos):
    mu, A, D, _ = movielens.build_instance(K=K, return_meta=True)
    H_cls = hardness.classical_hardness(mu)
    print(f"\n========= rho sweep at K = {K} (H_classical = {H_cls:.1f}) =========")
    print(f"{'rho':>6}  {'H_graph':>10}  {'H_cls/H_gr':>11}  "
          f"{'#competitive':>13}")
    for rho in rhos:
        H_gr = hardness.graph_hardness(mu, A, D, rho=rho)
        H_set, N_set = hardness.competitive_set(mu, A, D, rho=rho)
        ratio = H_cls / max(H_gr, 1e-9)
        print(f"{rho:>6.1f}  {H_gr:>10.2f}  {ratio:>10.2f}x  "
              f"{len(H_set):>5}/{len(H_set)+len(N_set)}")


def main():
    print("MovieLens-100K analytical pre-flight")
    print("====================================")
    rows = []
    for K in [20, 50, 100]:
        rows.append(report_one(K))
    report_rho_sweep(20, [1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0])

    print("\n========= summary =========")
    print(f"{'K':>5}  {'eps':>6}  {'gap_min':>7}  {'H_cls':>10}  "
          f"{'H_graph':>10}  {'edges':>7}")
    for r in rows:
        print(f"{r['K']:>5}  {r['eps']:>6.3f}  {r['gap_min']:>7.4f}  "
              f"{r['H_classical']:>10.2f}  {r['H_graph']:>10.2f}  "
              f"{r['n_edges']:>7}")
    print()
    print("Acceptance:")
    print("  - epsilon should be small enough that rho^*(eps) is in a")
    print("    reasonable range (~10-300)")
    print("  - smallest gap > 0 with 1/Delta^2 not absurd (< 1e6)")
    print("  - few/no isolated nodes (or accept that they will be")
    print("    handled like a Basic-TS arm by the regularizer)")


if __name__ == "__main__":
    main()
