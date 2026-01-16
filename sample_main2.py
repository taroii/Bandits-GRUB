import matplotlib.pyplot as plt
import numpy as np
import toml
import graph_algo
import graph_generator
import support_func


load_sample = False
read_config = True


def load_parameters(node_per_cluster=None, clusters=None):
    """

    Function to call graph_generator.py to provide graph data with the config setup from one of the following:
    1. Loads the sample graph setup from 'sample_setup.toml' (currently not compatible)
    2. Reading the setup from 'config.toml' and then generate graph.
    3. Generating graph from the default setup hardcodede in the function.

    Returns
    -------
    data : dictionary with graph related data (Degree, adjacency, means, total nodes).
    """
    module = toml.load('config.toml')
    config_data = module['system']
    #node_per_cluster = config_data['node_per_cluster']
    #clusters = config_data['clusters']
    p = config_data['p']
    q = config_data['q']
    graph_type = config_data['graph']

    cluster_means = node_per_cluster * clusters * np.ones(clusters) - node_per_cluster * np.arange(0, 1,
                                                                                                    1.0 / clusters)

    data = graph_generator.call_generator(node_per_cluster, clusters, p, cluster_means, graph_type, q=q)

    return data, node_per_cluster, clusters

def run_algo(GB, printer, cluster_size, i=0, t=0):
    """
    Run one of the algo picked from graph_algo.py

    Parameters
    ----------
    GB : One of the algorithms defined in graph_algo.py
    printer : Name tag for the algorithm (only required for printing)

    Returns
    -------
    Time_tracker_GB : Array with time marker when associated with arm elimination
    Lcluster_GB : Time marker when the best cluster was remaining
    Lnode_GB : Time marker when the last arm was remaining
    """

    Time_tracker_GB = []
    Lcluster_GB = []
    Lnode_GB = []

    flip = 1
    remainder = int(nodes)
    Time_tracker_GB.append(0)
    while (1):
        i += 1
        t += 1
        GB.play_round(1)

        if remainder != len(GB.remaining_nodes):
            for j in range(remainder - len(GB.remaining_nodes)):
                Time_tracker_GB.append(t)
            remainder = len(GB.remaining_nodes)

        if len(GB.remaining_nodes) <= cluster_size and flip:
            Lcluster_GB.append(t)
            flip = 0

        # Check for Thompson Sampling convergence or standard elimination
        if hasattr(GB, 'converged') and GB.converged:
            Lnode_GB.append(t)
            break
        if len(GB.remaining_nodes) == 1:
            Lnode_GB.append(t)
            break
    
    print("Node indices remaining : ", GB.remaining_nodes)
    print("Total time taken by algo ", printer, " : ", t)
    return Time_tracker_GB, Lnode_GB, Lcluster_GB

if __name__ == "__main__":
    """
    Sample code where we run 6 different algorithms from graph_algo.py.
    We change the number of arms (50, 100, 150, 200) and evaluate the performance of each algorithm on each setting.
    """

    arm_counts = [50, 100, 150, 200]
    clusters = 10
    results = {
        name: [] for name in ['Cyc', 'GB', 'GB_2', 'GB_sum', 'TS', 'Base']
    }

    for total_arms in arm_counts:
        node_per_cluster = total_arms // clusters
        print(f"\n=== Testing with {total_arms} arms ===")

        data, node_per_cluster, clusters = load_parameters(node_per_cluster, clusters)

        Degree = np.matrix(data['Degree'])
        Adj = np.matrix(data['Adj'])
        node_means = np.array(data['node_means'])
        nodes = data['nodes']

        # Set best arm to max * 1.2
        node_means[0] = max(node_means[1:]) * 1.2 

        eta = 1.0

        Cyc = graph_algo.CyclicAlgo(Degree, Adj, node_means, eta=eta)
        Time_tracker_Cyc, _, _ = run_algo(Cyc, printer="Cyc", cluster_size=node_per_cluster)
        results['Cyc'].append(Time_tracker_Cyc[-1])

        GB = graph_algo.MaxVarianceArmAlgo(Degree, Adj, node_means, eta=eta)
        Time_tracker_GB, _, _ = run_algo(GB, printer="GB", cluster_size=node_per_cluster)
        results['GB'].append(Time_tracker_GB[-1])

        GB_2 = graph_algo.MaxDiffVarAlgo(Degree, Adj, node_means, eta=eta, eps=0.0)
        Time_tracker_GB_2, _, _ = run_algo(GB_2, printer="GB_2", cluster_size=node_per_cluster)
        results['GB_2'].append(Time_tracker_GB_2[-1])

        GB_sum = graph_algo.OneStepMinSumAlgo(Degree, Adj, node_means, eta=eta, eps=0.0)
        Time_tracker_GB_sum, _, _ = run_algo(GB_sum, printer="GB_sum", cluster_size=node_per_cluster)
        results['GB_sum'].append(Time_tracker_GB_sum[-1])

        Base = graph_algo.NoGraphAlgo(Degree, Adj, node_means, eta=eta)
        Time_tracker_Base, _, _ = run_algo(Base, printer="Base", cluster_size=node_per_cluster)
        results['Base'].append(Time_tracker_Base[-1])

        TS = graph_algo.ThompsonSampling(Degree, Adj, node_means, eta=eta, delta=0.0001, q=0.01, eps=0.0)
        Time_tracker_TS, _, _ = run_algo(TS, printer="Thompson Sampling", cluster_size=node_per_cluster, t=TS.t)
        results['TS'].append(Time_tracker_TS[-1])

    # Plotting the results
    for name, times in results.items():                                                                                                     
        plt.plot(arm_counts, times, marker='o', label=name) 
                                                                                       
    plt.xlabel("Number of arms")                                                                                                            
    plt.ylabel("Time steps (arm pulls)")                                                                                                       
    plt.legend()                                                                                                                            
    plt.show()

