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

## Rebuttal-round experiments (NeurIPS 2026 author response)

These are additions for the author-response revision. They write their own
`.npz` files and **never overwrite** the committed figure data, so the
submitted numbers stay reproducible. Every runner is checkpointed per cell.

### Running the whole sweep on a server

```bash
conda activate thompson
pip install -r requirements.txt        # adds cvxpy + clarabel
bash run_server.sh                     # full sweep (20/10 seeds)
bash run_server.sh --pilot             # 5 seeds everywhere, much shorter
JOBS=32 bash run_server.sh             # cap concurrency (default: cores - 2)
```

`run_server.sh` launches 21 checkpointed jobs through a simple queue that
honours `JOBS`, then renders the tables and figures. Every process is pinned to
one BLAS thread — parallelism comes from running many processes, and
oversubscribed BLAS pools are what destabilised earlier shared-server runs (see
the note in `experiments/utils/runners.py`). **Re-running the script resumes**
rather than restarting, so it is safe to interrupt.

Progress:

```bash
tail -f experiments/outputs/logs/*.log
grep -h 't_med=' experiments/outputs/logs/grub_bias_*.log   # completed cells
```

Two things it deliberately does not do, both documented in
`REBUTTAL_FINDINGS.md`: it excludes GRUB from the MovieLens K=50 cell (§11),
and it runs the ε-sweep under both `ρ_diag` policies rather than picking one
(§2). Add GRUB back via `--algos` if you want the predicted cap confirmed.

Results from the laptop pilot that produced the numbers currently written up in
`REBUTTAL_FINDINGS.md` are preserved under
`experiments/outputs/mac_pilot/`. They are kept **out** of the resume path on
purpose: `.npz` checkpoints resume per cell, so leaving them in place would
produce tables mixing cells computed on two different BLAS backends, which
differ by 0.1–0.6% (see §13). The server run therefore starts cold and its
numbers are single-machine throughout.

| Task | Script | What it produces |
|---|---|---|
| 0 | `experiments/grub_audit.py` | Analytic decomposition of the GRUB radius: noise term vs. the four candidate bias terms, and `ln` vs `log2`. No bandit runs. |
| 0 | `experiments/grub_bias_sweep.py` | Reruns the GRUB cells of Figure 1 under `--bias {legacy,published,sqrt,oracle,none}`. |
| 0 | `experiments/grub_bias_plot.py` | Side-by-side figure + text table; cap-hits drawn hollow and never reported as stopping times. |
| 0 | `experiments/launch_grub_bias.sh` | Launches all (instance × bias) sweeps single-threaded. |
| 1 | `experiments/utils/tracing.py` | Per-run instrumentation: `t_eff` trajectories, per-arm pulls / `N_i^fb`, disagreement pairs, competitive-set membership under all three criteria in the draft, cap/correctness flags. Opt-in; leaves the RNG stream bit-identical. |
| 1 | `experiments/trace_runs.py` | Produces traces + the Q7 pooling-factor summary. |
| 2 | `experiments/utils/feedback_reg.py` | `FeedbackGraphBAI`: the full 2×2×2 design, estimator {`emp`,`reg`} × stop {`ts`,`ucb`} × pull {`cover`,`width`}. `reg` is the Laplacian-regularized estimator over side observations. |
| 2 | `experiments/fb_smooth.py` | The combined smoothness + feedback sweep. |
| 3 | `experiments/eps_sensitivity.py` | ε over/under-specification sweep with ρ set from Eq. (5). |
| 3 | `experiments/utils/adaptive_eps.py` | Residual smoothness estimator + doubling schedule. |
| 4a | `experiments/fb_scale.py` | Feedback sweep at larger `n`; reports `T/H_GF` and `T/(H_GF·L₂)` against `n` for all four (stop, pull) combinations. |
| 5 | `experiments/graph_params.py` | Influence-factor cross-check (5a), Corollary 13 sandwich (5b), and H_GF vs. the Russo et al. characteristic time (5c). |
| 6 | `experiments/utils/tas_fg.py` | TaS-FG (Russo, Song & Pacchiano, AISTATS 2025) as a live baseline. |

Two flags exist specifically so that a known algorithm/analysis mismatch is
covered from both sides rather than silently resolved:

* `algobase.AlgoBase(grub_bias=...)` — the GRUB bias convention. Default
  `'legacy'` reproduces the submitted numbers bit-for-bit.
* `graph_algo.GraphFeedbackTS(pull_scope=...)` — `'all'` is the implemented
  behaviour described in Appendix H.1 (`argmax` over every arm); `'pair'` is
  what Lemma 11 assumes (restricted to the disagreement pair).

`experiments/utils/hardness.py` gained `rho_var` (the Eq. (5) quantity
`σ₀²L₁(T)/ε²`). The pre-existing `rho_star` returns its **square root**; see
that function's docstring before using either.

### Two environment notes

* On macOS + Apple Accelerate, numpy 2.x emits spurious
  `RuntimeWarning: divide by zero encountered in matmul` for any matmul with
  dimension ≳32 (`np.ones((32,32)) @ np.ones(32)` reproduces it). Values are
  unaffected; do not suppress numpy warnings globally to hide it.
* The committed `.npz` files are not bit-reproducible off the original Ubuntu
  server: BLAS float drift over ~10⁶ Sherman–Morrison updates shifts GRUB
  stopping times by 0.1–0.6%. Compare medians at ~1% tolerance.

## Note on `misc/`

The `misc/` directory holds runners and plots for sanity checks and superseded experiments that are not referenced by the paper itself but are kept so that reviewers can spot-check additional regimes. These include a connectivity-disconnected variant of the SBM smoothness sweep (`mis_1.py`, superseded by `mis_2.py`), a $q$-sensitivity sweep (`q_sweep.py`), a graph-feedback evaluation on canonical graph families (`fb_structured.py`), a robustness check across MovieLens top-$k$ neighbour counts (`movielens_robustness.py`), and a $\rho$-sweep for GRUB on the synthetic chain (`grub_rho_sweep.py`). All are runnable in the same environment.

## Pre-trained models

This work studies non-parametric BAI algorithms (no learned model parameters). The saved `.npz` files in `experiments/outputs/` play the role typically served by pre-trained checkpoints: they are the raw stopping-time arrays from which every figure and table in the paper is regenerated. They are version-controlled with the rest of the repository.

## License

The code in this repository is released for review purposes only as part of the NeurIPS 2026 submission and is intended to be released under an open-source license (e.g. MIT) upon publication. Author and license details are withheld for double-blind review.
