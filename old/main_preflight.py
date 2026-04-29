"""main_preflight — analytical pre-flight for ``thm:main-graph``.

No TS runs.  Only computes the closed-form hardness numbers on
``instances.union_of_cliques_with_challenger`` for a sweep of K and prints

    K, |H|, |N|, H_graph, H_classical, max_N 1/Delta^2, ratio H_classical/H_graph,
    per-arm rho * J(i, G) * Delta_i^2 (sample of the smallest values, which
    decides which arms get classified competitive).

If H_graph stays roughly constant in K and H_classical grows roughly linearly
in K, the K-sweep experiment is worth running.  Otherwise the design is
broken and we redesign before burning compute.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.utils import instances, hardness  # noqa: E402


def diagnose(K, m=4, gap_chal=0.3, gap_decoy=1.5, rho=1.0):
    mu, A, D = instances.union_of_cliques_with_challenger(
        K, m=m, gap_chal=gap_chal, gap_decoy=gap_decoy)
    H_classical = hardness.classical_hardness(mu)
    H_graph = hardness.graph_hardness(mu, A, D, rho=rho)
    H_set, N_set = hardness.competitive_set(mu, A, D, rho=rho)

    a_star = int(np.argmax(mu))
    R = hardness.influence_factors(A, D)
    Delta = mu[a_star] - mu

    rows = []
    for i in range(K):
        if i == a_star:
            continue
        J = R[i, a_star]
        d2 = Delta[i] ** 2
        rho_J_d2 = rho * J * d2
        role = ('challenger' if i == 2 else
                'hub' if i == 1 else 'decoy')
        rows.append((i, role, Delta[i], J, rho_J_d2,
                     'comp' if i in H_set else 'non'))
    rows.sort(key=lambda r: r[4])

    max_N_inv_d2 = (max(1.0 / (Delta[i] ** 2) for i in N_set)
                    if N_set else 0.0)
    return {
        'K': K,
        'H_classical': H_classical,
        'H_graph': H_graph,
        'comp_size': len(H_set),
        'noncomp_size': len(N_set),
        'max_N_inv_d2': max_N_inv_d2,
        'rows': rows,
    }


def main():
    Ks = [20, 50, 100, 200, 400]
    delta = 1e-3
    log_inv_delta = np.log(1.0 / delta)
    print(f"# main_preflight  rho=1, m=4, gap_chal=0.3, gap_decoy=1.5, "
          f"delta={delta}, log(1/delta)={log_inv_delta:.2f}")
    print(f"\n{'K':>5} {'|H|':>4} {'|N|':>4} "
          f"{'H_graph':>8} {'H_class':>9} {'ratio':>7} "
          f"{'max_N 1/d^2':>11} "
          f"{'T_TS_pred':>10} {'T_Basic_pred':>12}")
    print('-' * 86)
    diag = []
    for K in Ks:
        d = diagnose(K)
        diag.append(d)
        T_TS = d['H_graph'] * log_inv_delta
        T_Basic = d['H_classical'] * log_inv_delta
        ratio = d['H_classical'] / max(d['H_graph'], 1e-9)
        print(f"{d['K']:>5} {d['comp_size']:>4} {d['noncomp_size']:>4} "
              f"{d['H_graph']:>8.2f} {d['H_classical']:>9.2f} "
              f"{ratio:>7.2f} "
              f"{d['max_N_inv_d2']:>11.3f} "
              f"{T_TS:>10.0f} {T_Basic:>12.0f}")

    print("\n# Per-arm rho*J(i,G)*Delta^2 classification at K=20 (sorted asc.; "
          "<= 1 -> competitive, > 1 -> non-competitive)")
    d = diagnose(20)
    print(f"\n{'arm':>4} {'role':>11} {'Delta':>6} "
          f"{'J(i,a*)':>8} {'rJd^2':>8} {'class':>6}")
    for r in d['rows'][:8]:
        print(f"{r[0]:>4} {r[1]:>11} {r[2]:>6.2f} {r[3]:>8.3f} {r[4]:>8.3f} {r[5]:>6}")
    print('  ...')
    for r in d['rows'][-3:]:
        print(f"{r[0]:>4} {r[1]:>11} {r[2]:>6.2f} {r[3]:>8.3f} {r[4]:>8.3f} {r[5]:>6}")

    # Acceptance
    print("\n# Acceptance for the design")
    H_graph_vals = [d['H_graph'] for d in diag]
    H_class_vals = [d['H_classical'] for d in diag]
    bounded = max(H_graph_vals) / min(H_graph_vals) <= 1.5
    grows = H_class_vals[-1] / H_class_vals[0] >= 5.0
    print(f"  [{'x' if bounded else ' '}] H_graph bounded across K  "
          f"(max/min = {max(H_graph_vals)/min(H_graph_vals):.2f})")
    print(f"  [{'x' if grows else ' '}] H_classical grows >= 5x across K  "
          f"(max/min = {H_class_vals[-1]/H_class_vals[0]:.2f})")


if __name__ == "__main__":
    main()
