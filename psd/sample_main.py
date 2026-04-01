"""
Sample main for PSD kernel setting (Kernel-TS-Explore).
Single-run comparison of Kernel-TS-Explore vs base algorithms.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import matplotlib.pyplot as plt
import numpy as np
import toml
import graph_algo
import graph_generator
from psd.psd_algo import KernelThompsonSampling


def load_parameters():
    config_path = os.path.join(os.path.dirname(__file__), 'config.toml')
    module = toml.load(config_path)
    config_data = module['system']
    node_per_cluster = config_data['node_per_cluster']
    clusters = config_data['clusters']
    p = config_data['p']
    q = config_data['q']
    graph_type = config_data['graph']

    cluster_means = node_per_cluster * clusters * np.ones(clusters) - node_per_cluster * np.arange(0, 1,
                                                                                                    1.0 / clusters)

    data = graph_generator.call_generator(node_per_cluster, clusters, p, cluster_means, graph_type, q=q)
    return data, node_per_cluster, clusters


def run_algo(GB, printer, cluster_size, nodes, i=0, t=0):
    Time_tracker = []
    Lnode = []

    flip = 1
    remainder = int(nodes)
    Time_tracker.append(0)
    while True:
        i += 1
        t += 1
        GB.play_round(1)

        if remainder != len(GB.remaining_nodes):
            for j in range(remainder - len(GB.remaining_nodes)):
                Time_tracker.append(t)
            remainder = len(GB.remaining_nodes)

        if hasattr(GB, 'converged') and GB.converged:
            Lnode.append(t)
            break
        if len(GB.remaining_nodes) == 1:
            Lnode.append(t)
            break

    print(f"Node indices remaining ({printer}): {GB.remaining_nodes}")
    print(f"Total time taken by {printer}: {t}")
    return Time_tracker, Lnode


if __name__ == "__main__":
    data, node_per_cluster, clusters = load_parameters()

    Degree = np.matrix(data['Degree'])
    Adj = np.matrix(data['Adj'])
    node_means = np.array(data['node_means'])
    nodes = data['nodes']

    config_path = os.path.join(os.path.dirname(__file__), 'config.toml')
    factor = toml.load(config_path)['system']['best_mean_factor']
    node_means[0] = max(node_means[1:]) * factor

    eta = 1.0

    Cyc = graph_algo.CyclicAlgo(Degree, Adj, node_means, eta=eta)
    GB = graph_algo.MaxVarianceArmAlgo(Degree, Adj, node_means, eta=eta)
    GB_2 = graph_algo.MaxDiffVarAlgo(Degree, Adj, node_means, eta=eta, eps=0.0)
    GB_sum = graph_algo.OneStepMinSumAlgo(Degree, Adj, node_means, eta=eta, eps=0.0)
    Base = graph_algo.NoGraphAlgo(Degree, Adj, node_means, eta=eta)
    TS = graph_algo.ThompsonSampling(Degree, Adj, node_means, eta=eta, delta=0.0001, q=0.01, eps=0.0)
    TS_K = KernelThompsonSampling(Degree, Adj, node_means, eta=eta, delta=0.0001, q=0.01, eps=0.0)

    Time_tracker_Cyc, _ = run_algo(Cyc, printer="Cyc", cluster_size=node_per_cluster, nodes=nodes)
    Time_tracker_GB, _ = run_algo(GB, printer="GB", cluster_size=node_per_cluster, nodes=nodes)
    Time_tracker_GB_2, _ = run_algo(GB_2, printer="GB_2", cluster_size=node_per_cluster, nodes=nodes)
    Time_tracker_GB_sum, _ = run_algo(GB_sum, printer="GB_sum", cluster_size=node_per_cluster, nodes=nodes)
    Time_tracker_Base, _ = run_algo(Base, printer="Base", cluster_size=node_per_cluster, nodes=nodes)
    Time_tracker_TS, _ = run_algo(TS, printer="TS-Explore", cluster_size=node_per_cluster, nodes=nodes, t=TS.t)
    Time_tracker_TS_K, _ = run_algo(TS_K, printer="Kernel-TS-Explore", cluster_size=node_per_cluster, nodes=nodes, t=TS_K.t)

    plt.plot(Time_tracker_GB_2, len(node_means)*np.ones(GB_2.dim) - range(len(Time_tracker_GB_2)), color='blue', marker='o', markersize=2, label='MVM', linewidth=2.0)
    plt.plot(Time_tracker_Cyc, len(node_means)*np.ones(Cyc.dim) - range(len(Time_tracker_Cyc)), marker='o', markersize=2, label='Cyclic', linewidth=2.0)
    plt.plot(Time_tracker_GB_sum, len(node_means)*np.ones(GB_sum.dim) - range(len(Time_tracker_GB_sum)), color='magenta', marker='^', markersize=4, label='JVM-N', linewidth=2.0)
    plt.plot(Time_tracker_GB, len(node_means)*np.ones(GB.dim) - range(len(Time_tracker_GB)), color='orange', marker='', markersize=4, label='JVM-O', linewidth=3.0)
    plt.plot(Time_tracker_TS, len(node_means)*np.ones(len(Time_tracker_TS)) - range(len(Time_tracker_TS)), color='red', marker='s', markersize=3, label='TS-Explore', linewidth=2.0, linestyle='--')
    plt.plot(Time_tracker_TS_K, len(node_means)*np.ones(len(Time_tracker_TS_K)) - range(len(Time_tracker_TS_K)), color='purple', marker='D', markersize=3, label='Kernel-TS-Explore', linewidth=2.0, linestyle='-.')
    plt.plot(Time_tracker_Base, len(node_means)*np.ones(Base.dim) - range(len(Time_tracker_Base)), color='green', marker='', markersize=4, label='No graph UCB', linewidth=2.0)

    plt.title("No. of remaining arms vs time steps (PSD Kernel)")
    plt.xlabel("Time steps")
    plt.ylabel("No. of remaining arms")
    plt.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0))
    plt.grid()
    plt.savefig(os.path.join(os.path.dirname(__file__), 'Figure_1.png'), dpi=150, bbox_inches='tight')
    plt.show()
