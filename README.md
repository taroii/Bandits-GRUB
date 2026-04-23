<!-- ABOUT THE PROJECT -->
## New Contributions: 
- ```sample_main.py``` now includes Thompson Sampling algorithm. It generates No. Remaining Arms vs Time Steps graph (figure 2 from paper)
- ```sample_main2.py``` generates Stopping time vs Number of arms on standard graph set up. 

## Setup

```
conda create -n thompson python=3.9
conda activate thompson
pip install -r requirements.txt
```

It's important to use an older version of python, since some code is depracated. 

## TODO: 
- Nothing!

## About The Project

We consider the problem of Best-Arm Identification (BAI) in the presence of imperfect side information in the form of a graph.

From a practical stand-point, here's why this is interesting:
* This approach helps bandit applications suffering from having too many sub-optimal choices. 
* No strict modelling assumption (like linear bandits, etc.) is needed.
* Superior experimental evidence backed by provably better sample complexity bounds.

This is the code base used for showing the superior experimental evidence.

## Core Problem

This repository addresses the challenge of finding the best "arm" (option) in a multi-armed bandit setting where:
- Multiple choices (arms) exist with unknown reward distributions
- Arms are connected through a graph structure representing similarities between them
- The objective is to identify the arm with the highest mean reward using minimal samples
- Graph structure provides side information that accelerates learning

## How It Works

1. **Graph Generation**: Create a graph where similar nodes (arms) are connected based on their properties
2. **Reward Modeling**: Each node has an unknown mean reward drawn from a Gaussian distribution
3. **Strategic Sampling**: Algorithms sample arms strategically using graph structure information
4. **Confidence Bounds**: As more samples are collected, confidence bounds shrink around mean estimates
5. **Elimination**: Arms are progressively eliminated when proven to be suboptimal
6. **Best Arm Identification**: Process continues until the best arm is identified with high confidence

The key insight is that connected nodes in the graph tend to have similar rewards. Therefore, sampling one node provides information about its neighbors, significantly reducing the overall sample complexity compared to standard bandit algorithms that ignore this structure.

## Repository Structure

### Key Components

#### 1. Graph-based Bandit Algorithms (`graph_algo.py`, `algobase.py`)
- **MaxVarianceArmAlgo**: Selects arms based on maximum variance/uncertainty
- **CyclicAlgo**: Cycles through arms following graph structure  
- **MaxDiffVarAlgo (JVM-O)**: Optimizes selection to reduce ensemble confidence width
- **OneStepMinSumAlgo (JVM-N)**: Minimizes sum of confidence widths across remaining arms
- **OneStepMinDetAlgo**: Minimizes determinant for optimal experimental design
- **NoGraphAlgo**: Baseline UCB algorithm without using graph information

#### 2. Algorithm Framework (`algobase.py`)
- Implements Laplacian-based mean estimation leveraging graph structure
- Uses UCB-style confidence bounds for arm elimination
- Sherman-Morrison formula for efficient matrix inverse updates
- Handles imperfect graph information with epsilon error bounds
- Tracks confidence widths and progressively eliminates suboptimal arms

#### 3. Graph Generation (`graph_generator.py`)
- Creates various test graph structures:
  - Stochastic Block Models (SBM)
  - Clustered graphs with configurable intra-cluster structures
- Generates node reward means based on cluster membership
- Supports graph subsampling for smaller-scale experiments
- Includes utilities for adding isolated nodes

#### 4. Support Functions (`support_func.py`)
- Gaussian reward generation with configurable variance
- Matrix operations including Sherman-Morrison inverse updates
- Cluster detection and jumping list generation for cyclic algorithms
- Mean vector optimization with Laplacian constraints
- Round function definitions for sampling schedules

## Getting Started

### System Requirements
The following packages in Python 3.6+ are required to run the simulations:
* numpy 1.19+
* scipy 1.5+
* matplotlib
* networkx 2.5+
* toml

Install dependencies:
```sh
pip install -r requirements.txt
```

### Running Base Experiment
The base file is `sample_main.py` can be executed by running:
```sh
python3 sample_main.py
```

For experiments with custom configurations:
```sh
python3 eps_best.py
```
 
### Default Configuration
* Total nodes = 101
* Total clusters = 10
* Nodes per cluster = 10
* 1 isolated optimal node
* Every cluster is a complete graph
* Best arm mean factor = 1.30 (optimal arm is 30% better than second best)

### Configuration Options

#### Graph Types
Apart from modifying node/cluster values in `config.toml`, the following graph structures are supported:
* **Tree graph**: Hierarchical structure within clusters
* **Star graph**: Central node connected to all others
* **Wheel graph**: Ring with central hub
* **Complete graph** (default): All nodes connected
* **Erdos-Renyi graph**: Random edges with probability p
* **Stochastic Block Model (SBM)**: Dense clusters, sparse inter-cluster (parameters p, q)
* **Barabasi-Albert graph**: Scale-free network (m=2)
* **Line graph**: Sequential node connections

#### Configuration Parameters (`config.toml`)
- `node_per_cluster`: Number of nodes in each cluster
- `clusters`: Total number of clusters
- `p`: Intra-cluster connection probability
- `q`: Inter-cluster connection probability (for SBM)
- `graph`: Graph type selection
- `best_mean_factor`: Multiplicative factor for optimal arm's mean

## Experimental Workflow

1. **Graph Setup**: Configure and generate graph structure based on parameters
2. **Mean Assignment**: Assign reward means to nodes based on cluster membership
3. **Algorithm Execution**: Run competing algorithms in parallel
4. **Performance Tracking**: Monitor arm elimination over time
5. **Visualization**: Plot number of remaining arms vs. time steps for comparison

<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.
 
