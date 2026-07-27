"""Task 5 -- graph parameter tables (no bandit runs).

5a  Influence factor and hardness table across canonical families, with a
    numerical cross-check of the three *different* identifications of the
    influence factor that appear in our draft, the rebuttal, and Thaker
    et al. (2022):

      thaker   J(j,G) = min_{i != j, i in C_j} r(i,j)^{-1}   = 1/max_i r(i,j)
               (Thaker et al. 2022, App. "Influence Factor", Definition;
               NOTE: a reciprocal, and independent of a*)
      pinv     J(i,G) = rho / [L_G^dagger]_ii
               (the identification asserted in our rebuttal)
      to_star  J(i,G) = r(i, a*)
               (what experiments/utils/hardness.py actually computes and
               what graph_hardness / competitive_set consume)

5b  Corrected Corollary 13 sandwich, checked numerically on non-uniform-gap
    instances:
      (rho_2(G) - 1) * Delta_max^-2  <=  H_GF
                                     <=  chi_bar(G) * Delta_min^-2
                                     <=  [n/(1+d_min)] * Delta_min^-2

5c  H_GF vs. the Russo-Song-Pacchiano (AISTATS 2025) characteristic time T*.

  python experiments/graph_params.py --n 20
  python experiments/graph_params.py --n 20 --which 5b
"""
from __future__ import annotations

import argparse
import itertools
import os
import sys

import numpy as np
import networkx as nx

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.utils import instances, hardness  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs')
os.makedirs(OUT, exist_ok=True)


# ---------------------------------------------------------------------------
# graph families
# ---------------------------------------------------------------------------

def _mats(G):
    n = G.number_of_nodes()
    A = np.zeros((n, n))
    for i, j in G.edges():
        A[i, j] = A[j, i] = 1.0
    return A, np.diag(A.sum(axis=1))


def families(n, gap=0.3, seed=0):
    """Yield (name, mu, A, D). Uniform-gap means unless noted."""
    out = []

    def uni(A, D, K=None):
        K = A.shape[0] if K is None else K
        mu = np.full(K, 1.0 - gap)
        mu[0] = 1.0
        return mu

    A, D = np.zeros((n, n)), np.zeros((n, n))
    out.append(('empty', uni(A, D), A, D))

    A, D = _mats(nx.complete_graph(n))
    out.append((f'complete K_{n}', uni(A, D), A, D))

    A, D = _mats(nx.star_graph(n - 1))
    out.append((f'star K_1,{n-1}', uni(A, D), A, D))

    A, D = _mats(nx.path_graph(n))
    out.append(('path', uni(A, D), A, D))

    # d-regular clique union: disjoint cliques of size d+1.
    for d in (3,):
        k = (n // (d + 1)) * (d + 1)
        G = nx.disjoint_union_all([nx.complete_graph(d + 1)
                                   for _ in range(k // (d + 1))])
        A, D = _mats(G)
        out.append((f'{d}-reg clique-union', uni(A, D), A, D))

    # d-regular expander (random regular graph).
    for d in (3, 4):
        if (n * d) % 2 == 0 and d < n:
            G = nx.random_regular_graph(d, n, seed=seed)
            A, D = _mats(G)
            out.append((f'{d}-reg random', uni(A, D), A, D))

    # Paper instances.
    mu, A, D = instances.clustered_chain(n, C=2, gap_step=gap)
    out.append(('clustered_chain', mu, A, D))

    mu, A, D = instances.sbm_phase_transition_connected()
    out.append(('SBM (K=31, conn.)', mu, A, D))

    mu, A, D = instances.ba_hub_optimal(n=n, m=2, gap=gap, seed=seed)
    out.append((f'BA hub-opt n={n}', mu, A, D))

    try:
        mu, A, D = instances.movielens_top_k(K=20, top_k_neighbors=5)
        out.append(('MovieLens K=20', mu, A, D))
    except Exception as e:  # dataset not cached
        print(f"  [skip MovieLens: {e}]", flush=True)

    return out


# ---------------------------------------------------------------------------
# 5a quantities
# ---------------------------------------------------------------------------

def resistance_matrix(A, D):
    """Full resistance-distance matrix; inf across components."""
    n = A.shape[0]
    L = D - A
    G = nx.from_numpy_array(A)
    R = np.full((n, n), np.inf)
    for comp in nx.connected_components(G):
        idx = np.array(sorted(comp))
        if len(idx) == 1:
            R[idx[0], idx[0]] = 0.0
            continue
        Ls = L[np.ix_(idx, idx)]
        Lp = np.linalg.pinv(Ls)
        d = np.diag(Lp)
        Rs = d[:, None] + d[None, :] - 2.0 * Lp
        R[np.ix_(idx, idx)] = np.maximum(Rs, 0.0)
    return R


def influence_thaker(A, D):
    """J(j,G) = 1 / max_{i != j, i in C_j} r(i,j); 0 if the component is
    a singleton (Thaker et al. 2022 definition, verbatim)."""
    R = resistance_matrix(A, D)
    n = A.shape[0]
    G = nx.from_numpy_array(A)
    J = np.zeros(n)
    comp_of = {}
    for ci, comp in enumerate(nx.connected_components(G)):
        for v in comp:
            comp_of[v] = ci
    comps = list(nx.connected_components(G))
    for j in range(n):
        comp = comps[comp_of[j]]
        others = [i for i in comp if i != j]
        if not others:
            J[j] = 0.0
        else:
            rmax = max(R[i, j] for i in others)
            J[j] = 1.0 / rmax if rmax > 0 else np.inf
    return J


def influence_pinv(A, D, rho=1.0):
    """J(i,G) = rho / [L_G^dagger]_ii (the rebuttal's identification)."""
    L = D - A
    Lp = np.linalg.pinv(L)
    d = np.diag(Lp)
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(np.abs(d) > 1e-12, rho / d, np.inf)


def influence_to_star(A, D, a_star):
    """J(i,G) = r(i, a*): what hardness.influence_factors feeds downstream."""
    return resistance_matrix(A, D)[:, a_star]


def two_packing_exact(A, max_n=40):
    """rho_2(G): max set of vertices with pairwise-disjoint closed
    neighbourhoods = max independent set in G^2.  Exact via MILP."""
    from scipy.optimize import milp, LinearConstraint, Bounds
    n = A.shape[0]
    if n > max_n:
        return None, 'skipped(n too large)'
    closed = (A + np.eye(n)) > 0
    # x_u + x_v <= 1 whenever N+(u) and N+(v) intersect (u != v).
    rows = []
    for u, v in itertools.combinations(range(n), 2):
        if (closed[u] & closed[v]).any():
            r = np.zeros(n)
            r[u] = r[v] = 1.0
            rows.append(r)
    cons = ([LinearConstraint(np.array(rows), -np.inf, 1.0)] if rows else [])
    res = milp(c=-np.ones(n), constraints=cons,
               integrality=np.ones(n), bounds=Bounds(0, 1))
    if not res.success:
        return None, f'MILP failed: {res.message}'
    return int(round(-res.fun)), 'exact'


def two_packing_fractional(A):
    """rho_2*(G): LP relaxation, max sum x s.t. sum_{u in N+(v)} x_u <= 1."""
    from scipy.optimize import linprog
    n = A.shape[0]
    closed = ((A + np.eye(n)) > 0).astype(float)
    res = linprog(-np.ones(n), A_ub=closed, b_ub=np.ones(n),
                  bounds=[(0, None)] * n, method='highs')
    if not res.success:
        return None
    return float(-res.fun)


def clique_cover_number(A, max_n=40):
    """chi_bar(G): min number of cliques covering V.  Exact MILP over
    maximal cliques for small n; greedy upper bound otherwise."""
    from scipy.optimize import milp, LinearConstraint, Bounds
    n = A.shape[0]
    G = nx.from_numpy_array(A)
    cliques = list(nx.find_cliques(G)) if n > 0 else []
    if n <= max_n and len(cliques) <= 20000:
        M = np.zeros((n, len(cliques)))
        for ci, c in enumerate(cliques):
            for v in c:
                M[v, ci] = 1.0
        # cover every vertex at least once
        cons = [LinearConstraint(M, 1.0, np.inf)]
        res = milp(c=np.ones(len(cliques)), constraints=cons,
                   integrality=np.ones(len(cliques)),
                   bounds=Bounds(0, 1))
        if res.success:
            return int(round(res.fun)), 'exact'
    # greedy: repeatedly take the largest clique among uncovered vertices
    uncovered = set(range(n))
    count = 0
    while uncovered:
        best = max((c for c in cliques),
                   key=lambda c: len(set(c) & uncovered), default=None)
        take = set(best) & uncovered if best else {next(iter(uncovered))}
        if not take:
            take = {next(iter(uncovered))}
        uncovered -= take
        count += 1
    return count, 'greedy(UB)'


def spectra(A, D):
    """(lambda_2, lambda_n, kappa) for L_G and the normalized Laplacian,
    plus the null-vector profile (v_0)_i^2 of L_G."""
    n = A.shape[0]
    L = D - A
    ev = np.sort(np.linalg.eigvalsh(L))
    lam2 = ev[1] if n > 1 else np.nan
    lamn = ev[-1]
    d = np.diag(D)
    dis = np.where(d > 0, 1.0 / np.sqrt(np.maximum(d, 1e-12)), 0.0)
    Kg = np.eye(n) - np.diag(dis) @ A @ np.diag(dis)
    evn = np.sort(np.linalg.eigvalsh(Kg))
    # null-vector profile: for a connected graph the L_G kernel is span(1),
    # so (v0)_i^2 = 1/n uniformly; for K_G it is D^{1/2}1 normalised.
    v0 = np.sqrt(np.maximum(d, 0.0))
    nv = np.linalg.norm(v0)
    prof = (v0 / nv) ** 2 if nv > 0 else np.full(n, np.nan)
    return dict(lam2=lam2, lamn=lamn,
                kappa=(lamn / lam2 if lam2 > 1e-12 else np.inf),
                nlam2=evn[1] if n > 1 else np.nan, nlamn=evn[-1],
                nkappa=(evn[-1] / evn[1] if evn[1] > 1e-12 else np.inf),
                v0sq_min=float(prof.min()), v0sq_max=float(prof.max()))


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------

def _fmt(x, w=8, p=3):
    if x is None:
        return f"{'--':>{w}s}"
    if isinstance(x, str):
        return f"{x:>{w}s}"
    if not np.isfinite(x):
        return f"{'inf':>{w}s}"
    return f"{x:>{w}.{p}f}"


def report_5a(fams, rho):
    print("\n" + "=" * 118)
    print("5a  INFLUENCE FACTOR CROSS-CHECK  (three identifications that "
          "appear in the draft / rebuttal / Thaker et al.)")
    print("=" * 118)
    print(f"{'family':22s} {'n':>4s} {'a*':>3s} | "
          f"{'thaker med':>10s} {'pinv med':>10s} {'r(i,a*) ':>10s} | "
          f"{'thaker=1/rmax':>13s} {'pinv==thaker?':>13s} "
          f"{'pinv==1/r(i,a*)?':>16s}")
    print("-" * 118)
    rows = []
    for name, mu, A, D in fams:
        n = A.shape[0]
        a_star = int(np.argmax(mu))
        if A.sum() == 0:
            print(f"{name:22s} {n:>4d} {a_star:>3d} | "
                  f"{'(no edges: all identifications degenerate)':>60s}")
            rows.append(dict(name=name, n=n))
            continue
        Jt = influence_thaker(A, D)
        Jp = influence_pinv(A, D, rho=rho)
        Js = influence_to_star(A, D, a_star)
        ft = Jt[np.isfinite(Jt)]
        fp = Jp[np.isfinite(Jp)]
        fs = Js[np.isfinite(Js)]
        # agreement tests (up to a global positive scale, since rho is free)
        def prop(a, b):
            m = np.isfinite(a) & np.isfinite(b) & (b > 0)
            if m.sum() < 2:
                return np.nan
            r = a[m] / b[m]
            return float(r.std() / max(abs(r.mean()), 1e-12))  # 0 => proportional
        agree_pt = prop(Jp, Jt)
        with np.errstate(divide='ignore'):
            inv_s = np.where(Js > 0, 1.0 / Js, np.inf)
        agree_ps = prop(Jp, inv_s)
        print(f"{name:22s} {n:>4d} {a_star:>3d} | "
              f"{_fmt(np.median(ft),10)} {_fmt(np.median(fp),10)} "
              f"{_fmt(np.median(fs),10)} | "
              f"{_fmt(np.median(ft),13)} "
              f"{_fmt(agree_pt,13)} {_fmt(agree_ps,16)}")
        rows.append(dict(name=name, n=n, Jt=Jt, Jp=Jp, Js=Js,
                         agree_pt=agree_pt, agree_ps=agree_ps))
    print("-" * 118)
    print("  'pinv==thaker?' / 'pinv==1/r(i,a*)?' show the coefficient of "
          "variation of the ratio;")
    print("  0.000 means the two are exactly proportional (identification "
          "valid up to the free rho scale), larger means they are different "
          "quantities.")
    return rows


def report_5a_struct(fams):
    print("\n" + "=" * 118)
    print("5a  STRUCTURAL / SPECTRAL / HARDNESS TABLE")
    print("=" * 118)
    print(f"{'family':22s} {'n':>4s} {'d_min':>5s} {'rho_2':>6s} "
          f"{'rho_2*':>7s} {'chi_bar':>8s} {'H_GF':>9s} {'H_cls':>9s} | "
          f"{'lam2':>7s} {'lam_n':>7s} {'kappa':>8s} {'nlam2':>7s} "
          f"{'nkappa':>8s} {'v0^2 max':>8s}")
    print("-" * 118)
    rows = []
    for name, mu, A, D in fams:
        n = A.shape[0]
        d_min = int(np.diag(D).min())
        p2, p2kind = two_packing_exact(A)
        p2f = two_packing_fractional(A)
        cc, cckind = clique_cover_number(A)
        try:
            hgf = hardness.graph_feedback_hardness(mu, A)
        except Exception:
            hgf = None
        hcls = hardness.classical_hardness(mu)
        sp = spectra(A, D)
        print(f"{name:22s} {n:>4d} {d_min:>5d} "
              f"{(str(p2) if p2 is not None else '--'):>6s} "
              f"{_fmt(p2f,7,2)} {str(cc):>8s} {_fmt(hgf,9,1)} "
              f"{_fmt(hcls,9,1)} | "
              f"{_fmt(sp['lam2'],7,3)} {_fmt(sp['lamn'],7,2)} "
              f"{_fmt(sp['kappa'],8,1)} {_fmt(sp['nlam2'],7,3)} "
              f"{_fmt(sp['nkappa'],8,1)} {_fmt(sp['v0sq_max'],8,4)}")
        rows.append(dict(name=name, n=n, d_min=d_min, rho2=p2,
                         rho2_frac=p2f, chi_bar=cc, chi_kind=cckind,
                         H_GF=hgf, H_cls=hcls, **sp))
    print("-" * 118)
    print("  rho_2 = 2-packing number (exact MILP), rho_2* = fractional "
          "relaxation (LP), chi_bar = clique-cover number,")
    print("  H_GF = covering-LP optimum, lam/nlam = combinatorial / "
          "normalized Laplacian spectra.")
    return rows


def report_5b(n, gap_seeds, rho, seed0=0):
    """Corollary 13 sandwich on NON-uniform-gap instances."""
    print("\n" + "=" * 112)
    print("5b  CORRECTED COROLLARY 13 SANDWICH  (non-uniform gaps)")
    print("      (rho_2 - 1)*Dmax^-2  <=  H_GF  <=  chi_bar*Dmin^-2  <=  "
          "[n/(1+d_min)]*Dmin^-2")
    print("=" * 112)
    print(f"{'family':22s} {'seed':>4s} {'LB':>9s} {'H_GF':>9s} "
          f"{'chi*Dm^-2':>10s} {'n/(1+dmin)':>11s} | {'LB<=H':>6s} "
          f"{'H<=chi':>7s} {'chi<=n/(1+d)':>12s}")
    print("-" * 112)
    rng = np.random.default_rng(seed0)
    viol = []
    base = [(nm, A, D) for nm, mu, A, D in families(n, gap=0.3, seed=seed0)
            if A.sum() > 0]
    for nm, A, D in base:
        K = A.shape[0]
        p2, _ = two_packing_exact(A)
        cc, _ = clique_cover_number(A)
        d_min = int(np.diag(D).min())
        if p2 is None:
            continue
        for s in range(gap_seeds):
            # random non-uniform gaps in [0.05, 1.0]
            mu = np.zeros(K)
            mu[0] = 1.0
            mu[1:] = 1.0 - rng.uniform(0.05, 1.0, size=K - 1)
            a_star = int(np.argmax(mu))
            gaps = mu[a_star] - mu
            gp = gaps[gaps > 0]
            Dmin, Dmax = gp.min(), gp.max()
            H = hardness.graph_feedback_hardness(mu, A)
            lb = (p2 - 1) / Dmax ** 2
            ub1 = cc / Dmin ** 2
            ub2 = (K / (1 + d_min)) / Dmin ** 2
            c1, c2, c3 = lb <= H + 1e-9, H <= ub1 + 1e-9, cc <= K / (1 + d_min) + 1e-9
            if not (c1 and c2 and c3):
                viol.append((nm, s, lb, H, ub1, ub2, c1, c2, c3))
            if s < 2:
                print(f"{nm:22s} {s:>4d} {lb:>9.1f} {H:>9.1f} "
                      f"{ub1:>10.1f} {ub2:>11.1f} | "
                      f"{str(c1):>6s} {str(c2):>7s} {str(c3):>12s}")
    print("-" * 112)
    if viol:
        print(f"  !! {len(viol)} VIOLATION(S) of the sandwich:")
        seen = set()
        for nm, s, lb, H, ub1, ub2, c1, c2, c3 in viol:
            which = ('LB<=H_GF' if not c1 else
                     'H_GF<=chi*Dmin^-2' if not c2 else
                     'chi_bar<=n/(1+d_min)')
            if (nm, which) in seen:
                continue
            seen.add((nm, which))
            print(f"     {nm:22s} fails {which:22s} "
                  f"(LB={lb:.1f} H={H:.1f} chi_ub={ub1:.1f} "
                  f"deg_ub={ub2:.1f})")
    else:
        print("  sandwich holds on every instance tested (no inversions)")

    # uniform-gap check (Table 1 regime): Dmin == Dmax
    print("\n  uniform-gap check (Table 1 regime, Delta_min = Delta_max):")
    for nm, A, D in base:
        K = A.shape[0]
        p2, _ = two_packing_exact(A)
        cc, _ = clique_cover_number(A)
        if p2 is None:
            continue
        d_min = int(np.diag(D).min())
        mu = np.full(K, 0.7)
        mu[0] = 1.0
        H = hardness.graph_feedback_hardness(mu, A)
        Dm = 0.3
        lb, ub1 = (p2 - 1) / Dm ** 2, cc / Dm ** 2
        ok = (lb <= H + 1e-9) and (H <= ub1 + 1e-9)
        print(f"    {nm:22s} LB={lb:>8.1f} H_GF={H:>8.1f} UB={ub1:>8.1f}  "
              f"{'ok' if ok else 'VIOLATED'}")
    return viol


def russo_characteristic_time(mu, A, lam=1.0, solver=None):
    """T*(nu) of Russo, Song & Pacchiano (AISTATS 2025), Gaussian case.

    Their Theorem 1 specialises, for nu_u = N(mu_u, lam^2) and a
    deterministic feedback graph G (G_{v,u} = 1 iff pulling v reveals u), to
    the convex program

        T* = inf_{omega in Delta(V)} max_{u != a*}
                 (1/m_u + 1/m_{a*}) * 2 lam^2 / Delta_u^2,
             s.t.  m = G^T omega

    so E[tau] >= T* * kl(delta, 1-delta).  Note T* charges for observing the
    *optimal* arm through the 1/m_{a*} term, whereas our covering LP H_GF
    imposes constraints only for u != a*.  Returned alongside is
    H_GF_matched = inf_omega max_{u != a*} 1/(m_u Delta_u^2), which is the
    same LP as hardness.graph_feedback_hardness written in normalised form
    (used here as an internal consistency check).
    """
    import cvxpy as cp
    mu = np.asarray(mu, float).flatten()
    K = len(mu)
    a_star = int(np.argmax(mu))
    gaps = mu[a_star] - mu
    closed = ((np.asarray(A, float) + np.eye(K)) > 0).astype(float)
    G = closed  # G[v, u] = 1 iff pulling v reveals u
    subopt = [u for u in range(K) if u != a_star]

    w = cp.Variable(K, nonneg=True)
    m = G.T @ w
    t = cp.Variable()
    cons = [cp.sum(w) == 1]
    for u in subopt:
        cons.append(2.0 * lam ** 2 / gaps[u] ** 2
                    * (cp.inv_pos(m[u]) + cp.inv_pos(m[a_star])) <= t)
    prob = cp.Problem(cp.Minimize(t), cons)
    try:
        prob.solve(solver=solver or cp.CLARABEL)
        T_star = float(prob.value) if prob.value is not None else None
    except Exception as e:
        return None, None, f'solve failed: {e}'

    # normalised covering LP, for cross-checking against hardness.py
    w2 = cp.Variable(K, nonneg=True)
    m2 = G.T @ w2
    t2 = cp.Variable()
    cons2 = [cp.sum(w2) == 1]
    for u in subopt:
        cons2.append(cp.inv_pos(m2[u]) / gaps[u] ** 2 <= t2)
    p2 = cp.Problem(cp.Minimize(t2), cons2)
    try:
        p2.solve(solver=solver or cp.CLARABEL)
        H_matched = float(p2.value) if p2.value is not None else None
    except Exception:
        H_matched = None
    return T_star, H_matched, 'ok'


def report_5c(fams, lam=1.0):
    print("\n" + "=" * 106)
    print("5c  H_GF  vs.  RUSSO-SONG-PACCHIANO (AISTATS 2025) CHARACTERISTIC "
          "TIME T*   (Gaussian, lambda=1)")
    print("=" * 106)
    print(f"{'family':22s} {'n':>4s} {'d_reg?':>7s} {'H_GF':>10s} "
          f"{'H_matched':>10s} {'T*':>10s} {'H_GF/T*':>9s} {'log n':>7s} "
          f"{'ratio/log n':>11s}")
    print("-" * 106)
    rows = []
    for name, mu, A, D in fams:
        n = A.shape[0]
        deg = np.diag(D)
        regular = 'yes' if deg.min() == deg.max() else 'no'
        try:
            hgf = hardness.graph_feedback_hardness(mu, A)
        except Exception:
            hgf = None
        T_star, H_matched, status = russo_characteristic_time(mu, A, lam=lam)
        if T_star is None or hgf is None:
            print(f"{name:22s} {n:>4d} {regular:>7s} {_fmt(hgf,10,1)} "
                  f"{'--':>10s} {'--':>10s} {'--':>9s} {'':>7s} "
                  f"{status:>11s}")
            continue
        ratio = hgf / T_star
        logn = np.log(n)
        print(f"{name:22s} {n:>4d} {regular:>7s} {hgf:>10.2f} "
              f"{_fmt(H_matched,10,2)} {T_star:>10.2f} {ratio:>9.3f} "
              f"{logn:>7.2f} {ratio/logn:>11.4f}")
        rows.append(dict(name=name, n=n, regular=regular, H_GF=hgf,
                         H_matched=H_matched, T_star=T_star, ratio=ratio))
    print("-" * 106)
    print("  H_matched = the same covering LP in normalised form "
          "(min_omega max_u 1/(m_u Delta_u^2)); it should equal H_GF.")
    print("  T* additionally charges for observing a* (the 1/m_{a*} term) and "
          "carries a factor 2 lambda^2, which H_GF omits.")
    reg = [r for r in rows if r['regular'] == 'yes']
    irr = [r for r in rows if r['regular'] == 'no']
    if reg:
        print(f"  regular graphs   : H_GF/T* in "
              f"[{min(r['ratio'] for r in reg):.3f}, "
              f"{max(r['ratio'] for r in reg):.3f}]")
    if irr:
        print(f"  irregular graphs : H_GF/T* in "
              f"[{min(r['ratio'] for r in irr):.3f}, "
              f"{max(r['ratio'] for r in irr):.3f}]")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=20)
    ap.add_argument('--rho', type=float, default=1.0,
                    help="rho in the pinv identification rho/[L^+]_ii")
    ap.add_argument('--gap-seeds', type=int, default=20,
                    help="5b: random non-uniform gap draws per family")
    ap.add_argument('--which', type=str, nargs='+',
                    default=['5a', '5a-struct', '5b', '5c'],
                    choices=['5a', '5a-struct', '5b', '5c'])
    args = ap.parse_args()

    print(__doc__)
    fams = families(args.n, gap=0.3)
    print(f"[graph_params] n={args.n}, {len(fams)} families")

    if '5a' in args.which:
        report_5a(fams, args.rho)
    if '5a-struct' in args.which:
        report_5a_struct(fams)
    if '5b' in args.which:
        report_5b(args.n, args.gap_seeds, args.rho)
    if '5c' in args.which:
        report_5c(fams)
    return 0


if __name__ == '__main__':
    sys.exit(main())
