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

    Cluster means are linearly spaced; after the isolated arm is inserted
    at index 0 its mean is ``best_factor * max(cluster_means)``.
    """
    rng_state = np.random.get_state()
    np.random.seed(seed)
    try:
        sizes = nodes_per_cluster * np.ones(n_clusters, dtype=int)
        probs = (p - q) * np.eye(n_clusters) + q * np.ones((n_clusters, n_clusters))
        G = nx.stochastic_block_model(sizes, probs, seed=seed)
        A, D = _mats_from_graph(G)
        cluster_means = nodes_per_cluster * n_clusters * np.ones(n_clusters) \
            - nodes_per_cluster * np.arange(0, 1, 1.0 / n_clusters)
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
