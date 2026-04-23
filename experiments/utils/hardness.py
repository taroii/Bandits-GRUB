"""Closed-form hardness measures for graph-bandit best-arm identification.

All calculators take the tuple ``(means, Adj, Degree)`` (numpy arrays) and
return a scalar hardness value.  The best arm is taken to be
``argmax(means)``.

Definitions follow "Efficient Thompson Sampling for Graph Bandits"
(Chang et al., 2026):

* :func:`classical_hardness` - sum_{i != *} 1 / Delta_i^2
* :func:`graph_hardness`     - H_graph from Theorem 4.1, with competitive /
  non-competitive split driven by the resistance-distance influence factor
  J(i, G) and a threshold ``rho * J(i, G) <= 1 / Delta_{i,c}^2``.
* :func:`graph_feedback_hardness` - H_GF from eq. (7) via an LP relaxation.
* :func:`rho_star`           - rho^*(eps) = sigma_0 sqrt(L1(T)) / eps.
* :func:`epsilon_hardness`   - H_eps from Theorem 4.4 (re-tuned rho).
* :func:`critical_epsilons`  - predicted phase-transition locations
  eps_i^* for each suboptimal arm.
* :func:`competitive_set`    - (H, N) partition at fixed rho.
* :func:`competitive_set_epsilon` - (H, N) partition as a function of eps.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linprog


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _best_and_gaps(means):
    means = np.asarray(means, dtype=float).flatten()
    a_star = int(np.argmax(means))
    Delta = means[a_star] - means
    return a_star, Delta


def influence_factors(Adj, Degree):
    """J(i, G) = L_G^+_{ii} + L_G^+_{a*, a*} - 2 L_G^+_{i, a*} on a graph.

    We return the full K x K resistance-distance matrix to avoid recomputation.
    """
    A = np.asarray(Adj, dtype=float)
    D = np.asarray(Degree, dtype=float)
    L = D - A
    L_pinv = np.linalg.pinv(L)
    diag = np.diag(L_pinv)
    R = diag[:, None] + diag[None, :] - 2.0 * L_pinv
    R = np.maximum(R, 0.0)  # clamp tiny negatives from numerical noise
    return R


def _gap_squared(means, a_star):
    """Delta_{i, c}^2 - we use the classical gap since the paper shares that notation."""
    Delta = means[a_star] - means
    return Delta ** 2


def _sigma0(sigma=1.0):
    return 2.0 * sigma * np.sqrt(14.0)


def _L1(K, T, delta):
    return np.log(12.0 * (K ** 2) * (max(T, 1.0) ** 2) / delta)


# ---------------------------------------------------------------------------
# Classical hardness
# ---------------------------------------------------------------------------

def classical_hardness(means):
    a_star, Delta = _best_and_gaps(means)
    H = 0.0
    for i, d in enumerate(Delta):
        if i == a_star:
            continue
        H += 1.0 / (d ** 2)
    return H


# ---------------------------------------------------------------------------
# H_graph (Theorem 4.1)
# ---------------------------------------------------------------------------

def graph_hardness(means, Adj, Degree, rho=1.0):
    """H_graph = sum_{competitive} 1/Delta^2 + max_{non-competitive} 1/Delta^2.

    An arm is competitive iff ``rho * J(i, G) <= 1 / Delta_{i, c}^2``
    (Thaker et al. 2022, Theorem C.1).  When the graph has no edges the
    influence factors diverge; we then fall back to classical hardness.
    """
    means = np.asarray(means, dtype=float).flatten()
    a_star, _ = _best_and_gaps(means)
    gap2 = _gap_squared(means, a_star)

    A = np.asarray(Adj, dtype=float)
    if A.sum() == 0:
        return classical_hardness(means)

    R = influence_factors(Adj, Degree)
    H_set, N_set = [], []
    for i in range(len(means)):
        if i == a_star:
            continue
        J_i = R[i, a_star]
        if not np.isfinite(J_i):
            N_set.append(i)
            continue
        if rho * J_i <= 1.0 / gap2[i]:
            H_set.append(i)
        else:
            N_set.append(i)

    H = sum(1.0 / gap2[i] for i in H_set)
    if N_set:
        H += max(1.0 / gap2[i] for i in N_set)
    return H


def competitive_set(means, Adj, Degree, rho=1.0):
    """Return (competitive_indices, non_competitive_indices) at fixed rho."""
    means = np.asarray(means, dtype=float).flatten()
    a_star, _ = _best_and_gaps(means)
    gap2 = _gap_squared(means, a_star)
    A = np.asarray(Adj, dtype=float)
    if A.sum() == 0:
        return [i for i in range(len(means)) if i != a_star], []

    R = influence_factors(Adj, Degree)
    H_set, N_set = [], []
    for i in range(len(means)):
        if i == a_star:
            continue
        J_i = R[i, a_star]
        if np.isfinite(J_i) and rho * J_i <= 1.0 / gap2[i]:
            H_set.append(i)
        else:
            N_set.append(i)
    return H_set, N_set


# ---------------------------------------------------------------------------
# Graph-feedback hardness (LP)
# ---------------------------------------------------------------------------

def graph_feedback_hardness(means, Adj):
    """H_GF from eq. (7) solved via linprog with method='highs'.

        minimize  sum_a tau_a
        subject to sum_{a: i in N+(a)} tau_a >= 1/Delta_i^2  for each i != *
                   tau_a >= 0.
    """
    means = np.asarray(means, dtype=float).flatten()
    a_star, _ = _best_and_gaps(means)
    gap2 = _gap_squared(means, a_star)
    K = len(means)
    A = np.asarray(Adj, dtype=float)
    closed = (A + np.eye(K)) > 0

    subopt = [i for i in range(K) if i != a_star]
    c = np.ones(K)
    A_ub = np.zeros((len(subopt), K))
    b_ub = np.zeros(len(subopt))
    for row, i in enumerate(subopt):
        for a in range(K):
            if closed[a, i]:
                A_ub[row, a] = -1.0
        b_ub[row] = -1.0 / gap2[i]
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=[(0, None)] * K, method='highs')
    if not res.success:
        raise RuntimeError(f"LP failed: {res.message}")
    return float(res.fun)


# ---------------------------------------------------------------------------
# rho^*(eps), H_eps, critical eps_i^*
# ---------------------------------------------------------------------------

def rho_star(epsilon, K, T_estimate, delta, sigma=1.0):
    """Optimal Laplacian weight for nominal smoothness ``epsilon``."""
    sigma0 = _sigma0(sigma)
    L1 = _L1(K, T_estimate, delta)
    return sigma0 * np.sqrt(L1) / max(epsilon, 1e-300)


def epsilon_hardness(means, Adj, Degree, epsilon, T_estimate, delta,
                     sigma=1.0, c0=4.0):
    """H_eps using Definition 4.3's eps-indexed competitive set.

    Arm i is competitive iff ``Delta_{i,c}^2 * J(i,G) <= c0 * sigma0 * eps *
    sqrt(L1(T))``.  Sum over competitive arms plus the max over non-competitive
    arms of ``1 / Delta_{i,c}^2``.
    """
    means = np.asarray(means, dtype=float).flatten()
    a_star, _ = _best_and_gaps(means)
    gap2 = _gap_squared(means, a_star)
    K = len(means)
    A = np.asarray(Adj, dtype=float)
    if A.sum() == 0:
        return classical_hardness(means)
    R = influence_factors(Adj, Degree)

    sigma0 = _sigma0(sigma)
    L1 = _L1(K, T_estimate, delta)
    thresh = c0 * sigma0 * epsilon * np.sqrt(L1)

    H_set, N_set = [], []
    for i in range(K):
        if i == a_star:
            continue
        J_i = R[i, a_star]
        if not np.isfinite(J_i):
            N_set.append(i)
            continue
        if gap2[i] * J_i <= thresh:
            H_set.append(i)
        else:
            N_set.append(i)

    H = sum(1.0 / gap2[i] for i in H_set)
    if N_set:
        H += max(1.0 / gap2[i] for i in N_set)
    return H


def competitive_set_epsilon(means, Adj, Degree, epsilon, T_estimate, delta,
                            sigma=1.0, c0=4.0):
    means = np.asarray(means, dtype=float).flatten()
    a_star, _ = _best_and_gaps(means)
    gap2 = _gap_squared(means, a_star)
    K = len(means)
    A = np.asarray(Adj, dtype=float)
    if A.sum() == 0:
        return [i for i in range(K) if i != a_star], []
    R = influence_factors(Adj, Degree)
    sigma0 = _sigma0(sigma)
    L1 = _L1(K, T_estimate, delta)
    thresh = c0 * sigma0 * epsilon * np.sqrt(L1)
    H_set, N_set = [], []
    for i in range(K):
        if i == a_star:
            continue
        J_i = R[i, a_star]
        if np.isfinite(J_i) and gap2[i] * J_i <= thresh:
            H_set.append(i)
        else:
            N_set.append(i)
    return H_set, N_set


def critical_epsilons(means, Adj, Degree, T_estimate, delta,
                      sigma=1.0, c0=4.0):
    """eps_i^* = Delta_{i,c}^2 * J(i,G) / (c0 * sigma0 * sqrt(L1(T))) for each
    suboptimal arm.  Arms exit the competitive set as epsilon drops below
    their eps_i^*.
    """
    means = np.asarray(means, dtype=float).flatten()
    a_star, _ = _best_and_gaps(means)
    gap2 = _gap_squared(means, a_star)
    K = len(means)
    A = np.asarray(Adj, dtype=float)
    if A.sum() == 0:
        return {i: float('inf') for i in range(K) if i != a_star}
    R = influence_factors(Adj, Degree)
    sigma0 = _sigma0(sigma)
    L1 = _L1(K, T_estimate, delta)
    out = {}
    for i in range(K):
        if i == a_star:
            continue
        J_i = R[i, a_star]
        if not np.isfinite(J_i) or J_i == 0:
            out[i] = float('inf')
        else:
            out[i] = gap2[i] * J_i / (c0 * sigma0 * np.sqrt(L1))
    return out
