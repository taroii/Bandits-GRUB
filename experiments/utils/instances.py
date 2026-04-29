"""Standardized test-instance builders for the experiments.

Every builder returns a (means, Adj, Degree) triple.  The best arm is always
at index 0 unless noted.  These wrap :func:`graph_generator.call_generator`
and related helpers with deterministic naming.
"""
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import numpy as np
import networkx as nx

import graph_generator as gg  # noqa: E402


def _mats_from_graph(G):
    n = G.number_of_nodes()
    A = np.zeros((n, n))
    for i, j in G.edges():
        A[i, j] = 1.0
        A[j, i] = 1.0
    D = np.diag(A.sum(axis=1))
    return A, D


def sbm_standard(n_clusters=10, nodes_per_cluster=10, p=0.9, q=0.0,
                 best_factor=1.3, seed=0):
    """SBM with one isolated best arm at index 0.

    Cluster means are linearly spaced in [0.1, 1.0] (on the same scale as
    the Gaussian reward noise, sigma=1, so gaps are comparable to the
    noise level and the instance is actually hard).  After the isolated
    arm is inserted at index 0 its mean is ``best_factor * max(cluster_means)``.
    """
    rng_state = np.random.get_state()
    np.random.seed(seed)
    try:
        sizes = nodes_per_cluster * np.ones(n_clusters, dtype=int)
        probs = (p - q) * np.eye(n_clusters) + q * np.ones((n_clusters, n_clusters))
        G = nx.stochastic_block_model(sizes, probs, seed=seed)
        A, D = _mats_from_graph(G)
        cluster_means = np.linspace(1.0, 0.1, n_clusters)
        mu = np.zeros(n_clusters * nodes_per_cluster)
        for c in range(n_clusters):
            mu[c * nodes_per_cluster:(c + 1) * nodes_per_cluster] = cluster_means[c]
        A, D, mu = gg.one_off_setup(A, D, mu, best_factor * cluster_means.max())
        A[0, 1] = 1.0
        A[1, 0] = 1.0
        D[0, 0] = 1.0
        D[1, 1] += 1.0
    finally:
        np.random.set_state(rng_state)
    return np.asarray(mu, dtype=float), np.asarray(A, dtype=float), np.asarray(D, dtype=float)


def erdos_renyi(n=50, p=0.2, gap=0.3, seed=0):
    """ER graph with best arm at index 0 (mean 1.0), rest at 1-gap."""
    G = nx.erdos_renyi_graph(n, p, seed=seed)
    A, D = _mats_from_graph(G)
    mu = np.full(n, 1.0 - gap)
    mu[0] = 1.0
    return mu, A, D


def barabasi_albert(n=100, m=2, gap=0.3, seed=0):
    G = nx.barabasi_albert_graph(n, m, seed=seed)
    A, D = _mats_from_graph(G)
    mu = np.full(n, 1.0 - gap)
    mu[0] = 1.0
    return mu, A, D


def complete_graph(n=50, gap=0.3, seed=0):
    G = nx.complete_graph(n)
    A, D = _mats_from_graph(G)
    mu = np.full(n, 1.0 - gap)
    mu[0] = 1.0
    return mu, A, D


def empty_graph(n=50, gap=0.3, seed=0):
    A = np.zeros((n, n))
    D = np.zeros((n, n))
    mu = np.full(n, 1.0 - gap)
    mu[0] = 1.0
    return mu, A, D


def union_of_cliques_with_challenger(K, m=4, gap_chal=0.3, gap_decoy=1.5,
                                     mu_best=1.0, seed=0):
    """K-arm instance designed for ``thm:main-graph`` K-scaling.

    Layout:
        index 0 : best arm        (mean = mu_best)
        index 1 : hub             (gap = gap_decoy)
        index 2 : challenger      (gap = gap_chal)
        indices 3..K-1 : decoys, partitioned into cliques of size ~m,
                         each clique has one node bridged to the hub.

    Edges:
        best — hub
        hub  — challenger
        hub  — clique-bridge node (one per clique)
        clique : complete graph on its m nodes

    With m=4, gap_chal=0.3, gap_decoy=1.5, rho=1 the hardness numbers are

        H_graph     ≈ 1/gap_chal^2 + 1/gap_decoy^2     (constant in K)
        H_classical = 1/gap_chal^2 + (K-2)/gap_decoy^2 (linear in K)

    so the *visible* signature is: TS-Explore plateaus while Basic TS grows
    linearly in K. ``seed`` is accepted but unused (deterministic graph).
    """
    if K < 4:
        raise ValueError(f"K={K} too small; need K >= 4")
    n_decoy = K - 3
    n_cliques = max(1, n_decoy // m)
    sizes = [m] * n_cliques
    sizes[-1] += n_decoy - sum(sizes)

    A = np.zeros((K, K))
    A[0, 1] = A[1, 0] = 1.0
    A[1, 2] = A[2, 1] = 1.0
    idx = 3
    for sz in sizes:
        for i in range(idx, idx + sz):
            for j in range(i + 1, idx + sz):
                A[i, j] = A[j, i] = 1.0
        A[1, idx] = A[idx, 1] = 1.0
        idx += sz

    D = np.diag(A.sum(axis=1))
    mu = np.full(K, mu_best - gap_decoy)
    mu[0] = mu_best
    mu[2] = mu_best - gap_chal
    return mu, A, D


def sbm_phase_transition(seed=0):
    """Instance for Experiment 3.

    5 clusters x 6 nodes with cluster means linearly spaced in [0.1, 0.9],
    plus one isolated best arm at index 0 (mean 1.0).  Designed so that the
    critical smoothness values epsilon_i^star are well separated.
    """
    n_clusters = 5
    per = 6
    G = nx.stochastic_block_model(per * np.ones(n_clusters, dtype=int),
                                  0.9 * np.eye(n_clusters) + 0.0,
                                  seed=seed)
    A, D = _mats_from_graph(G)
    cluster_means = np.array([0.9, 0.7, 0.5, 0.3, 0.1])
    mu = np.zeros(n_clusters * per)
    for c in range(n_clusters):
        mu[c * per:(c + 1) * per] = cluster_means[c]
    A, D, mu = gg.one_off_setup(A, D, mu, 1.0)
    A[0, 1] = 1.0
    A[1, 0] = 1.0
    D[0, 0] = 1.0
    D[1, 1] += 1.0
    return np.asarray(mu, dtype=float), np.asarray(A, dtype=float), np.asarray(D, dtype=float)
