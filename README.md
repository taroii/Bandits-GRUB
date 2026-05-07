# Thompson Sampling for Pure Exploration in Graph-Structured Bandits

This repository is the official implementation accompanying the paper *"Thompson Sampling for Pure Exploration in Graph-Structured Bandits"* (NeurIPS 2026 anonymous submission). It contains the source for the proposed algorithms, the runners that reproduce every experiment in the paper, and the saved raw stopping-time data.

## Overview

We study the best-arm identification (BAI) problem in stochastic bandits where similarities between arms are described by a graph. The graph enters either through a *smoothness regularizer* on the reward vector (the graph-smooth setting) or through *side observations* (the graph-feedback setting). For each setting we develop a Thompson-sampling–based algorithm with a fixed-confidence sample-complexity bound that improves on the classical hardness $\sum_i \Delta_i^{-2}$ by replacing it with a graph-aware quantity.

## Requirements

The experiments run on CPU only and have no external data dependency at install time (the MovieLens-100K dataset is downloaded automatically on first use of the corresponding runner).

To install dependencies:

```setup
conda create -n thompson python=3.11
conda activate thompson
pip install -r requirements.txt
```

The pinned versions reflect the environment used to produce the figures in the paper; later versions of the listed packages should also work.

## Running the experiments

The paper-experiment pipeline lives under `experiments/`. Each experiment has a *runner* that executes the sweep and writes a `.npz` result file to `experiments/outputs/`, and a *plot* script that reads the `.npz` and renders the figure. Runners are checkpointed: rerunning a runner resumes from its last completed cell, and `--fresh` ignores the checkpoint and starts over. Every runner accepts `--quick` for a smoke-test sweep (much smaller seed/sweep counts) to verify the end-to-end pipeline before launching the full multi-hour sweep.

| Experiment                                              | Runner(s)                                                         | Plot / table script                |
|---------------------------------------------------------|-------------------------------------------------------------------|------------------------------------|
| Synthetic chain $K$-sweep (graph-smooth, two-cluster)   | `experiments/main_2.py`                                           | `experiments/fig1_plot.py`         |
| MovieLens-100K $\rho$-sweep (graph-smooth, real)        | `experiments/movielens_1.py`                                      | `experiments/fig1_plot.py`         |
| Erdős–Rényi density sweep (graph feedback, headline)    | `experiments/fb_1.py`                                             | `experiments/fb_1_plot.py`         |
| 2×2 stop-rule × pull-rule ablation (graph feedback)     | `experiments/fb_ablation.py`                                      | `experiments/fb_1_plot.py`         |
| Barabási–Albert kernel comparison ($L_G$ vs. $K_G$)     | `experiments/kernel_1.py`                                         | `experiments/kernel_1_plot.py`     |
| Connected-SBM smoothness asymptotics                    | `experiments/mis_2.py`                                            | (numbers used directly in the LaTeX source) |

To reproduce all paper figures from scratch, run every runner once and then every plot script:

```reproduce
# Runners (long-running; checkpointed; resume on rerun).
python experiments/main_2.py
python experiments/movielens_1.py
python experiments/fb_1.py
python experiments/fb_ablation.py
python experiments/kernel_1.py
python experiments/mis_2.py

# Plot scripts (fast; read .npz files and write PDFs/PNGs).
python experiments/fig1_plot.py
python experiments/fb_1_plot.py
python experiments/kernel_1_plot.py
```

Each plot script writes both a `.pdf` (vector, included by the LaTeX source) and a `.png` (raster preview) under `experiments/outputs/`. The shared paper-figure style is defined in `experiments/utils/plotting.py`. The `.npz` files are precomputed and shipped in `experiments/outputs/`, so plot regeneration is seconds.

## Results

The headline empirical findings of the paper, which the runners + plot scripts above reproduce verbatim from the saved `.npz` files:

| Setting                                                | Result                                                                                                                                                                            |
|--------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Synthetic chain, $K\in\{10,20,50,100,200\}$, $\rho=100$ | TS-Explore stopping time *decreases* with $K$ (cluster pooling); Basic TS, KL-LUCB, GRUB grow linearly. TS-Explore is $>10\times$ faster than Basic TS at $K=100$.                |
| MovieLens-100K, $K=20$, $\rho \in [1,10^3]$            | TS-Explore at $\rho=10^3$ is more than an order of magnitude faster than Basic TS and below the strongest non-graph baseline KL-LUCB.                                              |
| Erdős–Rényi $G(20,p)$, graph feedback                  | TS-Explore-GF and UCB+cover both shrink by $\sim 10\times$ as $p$ increases; the cover-pair pull rule is the dominant empirical effect, isolated by the $2{\times}2$ ablation.    |
| Barabási–Albert $n=50$, hub-optimal                    | Normalized Laplacian $K_G$ degrades far more gracefully than combinatorial $L_G$ at large $\rho$ and improves over $L_G$ by a factor of $1.12$ ($95\%$ bootstrap CI $[1.05, 1.19]$). |
| Connected SBM smoothness asymptotics                   | TS-Explore at $\rho \ge \rho_{\mathrm{var}}(\varepsilon)$ drops $11.3\times$ below the un-tuned baselines at $\varepsilon = 10^{-2.5}$.                                            |

All entries are medians over the seeds reported in the paper (20 seeds for main-body cells, 5–10 seeds for appendix cells), with $25$–$75$ interquartile ranges shaded in the figures.

## Note on `misc/`

The `misc/` directory holds runners and plots for sanity checks and superseded experiments that are not referenced by the paper itself but are kept so that reviewers can spot-check additional regimes. These include a connectivity-disconnected variant of the SBM smoothness sweep (`mis_1.py`, superseded by `mis_2.py`), a $q$-sensitivity sweep (`q_sweep.py`), a graph-feedback evaluation on canonical graph families (`fb_structured.py`), a robustness check across MovieLens top-$k$ neighbour counts (`movielens_robustness.py`), and a $\rho$-sweep for GRUB on the synthetic chain (`grub_rho_sweep.py`). All are runnable in the same environment.

## Pre-trained models

This work studies non-parametric BAI algorithms (no learned model parameters). The saved `.npz` files in `experiments/outputs/` play the role typically served by pre-trained checkpoints: they are the raw stopping-time arrays from which every figure and table in the paper is regenerated. They are version-controlled with the rest of the repository.

## License

The code in this repository is released for review purposes only as part of the NeurIPS 2026 submission and is intended to be released under an open-source license (e.g. MIT) upon publication. Author and license details are withheld for double-blind review.
