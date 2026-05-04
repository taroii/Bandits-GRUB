<!-- ABOUT THE PROJECT -->
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

## Reproducing the paper experiments

The paper-experiment pipeline lives under `experiments/`. Each experiment has two scripts: a **runner** that executes the sweep and writes a `.npz` result file to `experiments/outputs/`, and a **plot** script that reads the `.npz` and renders the figure.

**Runners** (long-running; resume from a checkpoint if rerun):
- `experiments/main_2.py` — clustered-chain K-sweep (graph-smooth, synthetic; Thm. 3.4)
- `experiments/movielens_1.py` — MovieLens-100K rho-sweep (graph-smooth, real)
- `experiments/fb_1.py` — Erdos-Renyi density sweep (graph feedback; Thm. 3.10)
- `experiments/mis_1.py` — SBM smoothness asymptotics (Cor. 3.7)
- `experiments/kernel_1.py` — Barabasi-Albert kernel comparison (Thm. 3.11)
- `experiments/movielens_ablations.py` — graph-construction robustness
- `experiments/main_1.py` — agreement-vs-elimination tightness (appendix)

Each runner takes `--quick` for a smoke-test sweep and `--fresh` to ignore an existing checkpoint.

**Plot scripts** (fast; read `.npz` and write figures to `experiments/outputs/`):
- `experiments/fig1_plot.py` — combined paper Figure 1 (synthetic + MovieLens)
- `experiments/fb_1_plot.py` — paper Figure 2 (graph feedback)
- `experiments/mis_1_plot.py`, `kernel_1_plot.py`, `movielens_ablations_plot.py`, `main_1_plot.py` — appendix figures
- `experiments/main_2_plot.py`, `movielens_1_plot.py` — single-panel variants of the Fig. 1 components

Every plot script writes the figure as both a `.pdf` (vector, used by the LaTeX paper) and a `.png` (raster preview) under the same base name; the shared paper style lives in `experiments/utils/plotting.py`.

### Compiling the paper

With all `.pdf` figures present in `experiments/outputs/`:
```sh
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.
 
