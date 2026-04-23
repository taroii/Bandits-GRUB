"""
Sample main 2 for PSD kernel setting (Kernel-TS-Explore).
Sweeps over arm counts comparing Kernel-TS-Explore vs base algorithms.
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


def load_parameters(node_per_cluster, clusters):
    config_path = os.path.join(os.path.dirname(__file__), 'config.toml')
    module = toml.load(config_path)
    config_data = module['system']
    p = config_data['p']
    q = config_data['q']
    graph_type = config_data['graph']

    cluster_means = node_per_cluster * clusters * np.ones(clusters) - node_per_cluster * np.arange(0, 1,
                                                                                                    1.0 / clusters)

    data = graph_generator.call_generator(node_per_cluster, clusters, p, cluster_means, graph_type, q=q)
    return data, node_per_cluster, clusters


def run_algo(GB, printer, nodes, t=0):
    remainder = int(nodes)
    while True:
        t += 1
        GB.play_round(1)

        if hasattr(GB, 'converged') and GB.converged:
            break
        if len(GB.remaining_nodes) == 1:
            break

    print(f"  {printer}: t={t}, remaining={GB.remaining_nodes}")
    return t


if __name__ == "__main__":
    arm_counts = [50, 100, 150, 200]
    clusters = 10
    results = {name: [] for name in ['Cyc', 'GB', 'GB_2', 'GB_sum', 'Base', 'TS', 'TS_K']}

    for total_arms in arm_counts:
        node_per_cluster = total_arms // clusters
        print(f"\n=== Testing with {total_arms} arms ===")

        data, node_per_cluster, clusters = load_parameters(node_per_cluster, clusters)

        Degree = np.matrix(data['Degree'])
        Adj = np.matrix(data['Adj'])
        node_means = np.array(data['node_means'])
        nodes = data['nodes']

        node_means[0] = max(node_means[1:]) * 1.2

        eta = 1.0

        Cyc = graph_algo.CyclicAlgo(Degree, Adj, node_means, eta=eta)
        results['Cyc'].append(run_algo(Cyc, printer="Cyc", nodes=nodes))

        GB = graph_algo.MaxVarianceArmAlgo(Degree, Adj, node_means, eta=eta)
        results['GB'].append(run_algo(GB, printer="GB", nodes=nodes))

        GB_2 = graph_algo.MaxDiffVarAlgo(Degree, Adj, node_means, eta=eta, eps=0.0)
        results['GB_2'].append(run_algo(GB_2, printer="MVM", nodes=nodes))

        GB_sum = graph_algo.OneStepMinSumAlgo(Degree, Adj, node_means, eta=eta, eps=0.0)
        results['GB_sum'].append(run_algo(GB_sum, printer="GB_sum", nodes=nodes))

        Base = graph_algo.NoGraphAlgo(Degree, Adj, node_means, eta=eta)
        results['Base'].append(run_algo(Base, printer="No Graph UCB", nodes=nodes))

        TS = graph_algo.ThompsonSampling(Degree, Adj, node_means, eta=eta, delta=0.0001, q=0.01, eps=0.0)
        results['TS'].append(run_algo(TS, printer="TS-Explore", nodes=nodes, t=TS.t))

        TS_K = KernelThompsonSampling(Degree, Adj, node_means, eta=eta, delta=0.0001, q=0.01, eps=0.0)
        results['TS_K'].append(run_algo(TS_K, printer="Kernel-TS-Explore", nodes=nodes, t=TS_K.t))

    labels = {'Cyc': 'Cyclic', 'GB': 'JVM-O', 'GB_2': 'MVM', 'GB_sum': 'JVM-N', 'Base': 'No Graph UCB', 'TS': 'TS-Explore', 'TS_K': 'Kernel-TS-Explore'}
    for name, times in results.items():
        plt.plot(arm_counts, times, marker='o', label=labels[name])

    plt.xlabel("Number of arms")
    plt.ylabel("Time steps (arm pulls)")
    plt.title("Sample Complexity vs Number of Arms (PSD Kernel)")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(os.path.dirname(__file__), 'Figure_2.png'), dpi=150, bbox_inches='tight')
    plt.show()
