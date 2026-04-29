# Experimental Validation Plan: Efficient Thompson Sampling for Graph Bandits

This document specifies experiments to validate the theoretical results of the paper. Each experiment targets a specific theorem. Work through these in the order given — infrastructure first, then experiments in dependency order.

---

## Context

**Paper:** "Efficient Thompson Sampling for Graph Bandits" — develops TS-based pure-exploration algorithms across four settings:
- §4.1 Graph-smooth bandits (Laplacian regularization) — Theorem 4.1 (`thm:main-graph`)
- §4.2 Misspecification and smoothness asymptotics — Theorem 4.4 (`thm:main-mis`) and Corollary 4.5 (`cor:eps-limit`)
- §4.3 Graph feedback — Theorem 4.8 (`thm:main-fb`)
- §4.4 General PSD kernels (normalized Laplacian) — Theorem 4.9 (`thm:kernel-ts`)

**Important update in the latest revision:** §4.2 was reframed. It is no longer about "graceful degradation" when the true smoothness exceeds the nominal one. The new framing is **optimal tuning**: since $\rho$ and $\varepsilon$ enter the confidence radius only through the product $\rho\varepsilon$, misspecification of $\varepsilon$ can be absorbed by adjusting $\rho$. The main result studies the asymptotic regime $\varepsilon \to 0$ with $\rho^\star(\varepsilon) = \sigma_0\sqrt{L_1(T)}/\varepsilon$, and shows that the competitive set shrinks until $H_\varepsilon \to \max_{i \neq a^\star} \Delta_{i,c}^{-2}$. Corollary 4.5 predicts **phase transitions**: each suboptimal arm $i$ has a critical smoothness level $\varepsilon_i^\star = \Delta_{i,c}^2 \mathcal{J}(i,G) / (c_0 \sigma_0 \sqrt{L_1(T)})$, and arms exit the competitive set in order of decreasing $\varepsilon_i^\star$.

**Existing code:**
- `graph_algo.py` — contains `ThompsonSampling` class (graph-smooth Laplacian variant), UCB baselines (`MaxVarianceArmAlgo`, `MaxDiffVarAlgo`, `OneStepMinSumAlgo`, `CyclicAlgo`, `NoGraphAlgo`), and `BasicThompsonSampling` (no-graph TS)
- `algobase.py` — base class with Laplacian-regularized estimator, Sherman–Morrison updates, arm elimination
- `graph_generator.py` — SBM, ER, BA, complete, star, wheel, line, tree graph generators
- `support_func.py` — Gaussian rewards, matrix utilities
- `sample_main.py` — produces Figure 1 (single-run elimination curve)
- `sample_main2.py` — produces Figure 2 (stopping time vs number of arms)

**Known issues with existing experiments:**
1. Single-run results (no seed variance / confidence intervals)
2. `ThompsonSampling.play_round` uses `np.linalg.inv` each round — should use Sherman–Morrison for scalability
3. TS algorithm doesn't produce intermediate elimination events, so Figure 1 shows it as a flat vertical line (misleading)
4. No experiments exist for graph feedback, kernel variant, or misspecification/asymptotics — three of the four theorems are untested
5. The current `ThompsonSampling` class accepts `eps` but hardcodes a particular relationship between `eta` (the regularization scale, which the paper calls $\rho$) and `eps` (which plays the role of $\varepsilon$). The new §4.2 requires us to sweep over $\varepsilon$ with $\rho = \rho^\star(\varepsilon)$, so the algorithm needs to expose $\rho$ and $\varepsilon$ as independent knobs.

---

## Phase 0: Shared Infrastructure

Build all of this before touching individual experiments. Each experiment depends on it.

### 0.1 Create `experiments/` directory structure

```
experiments/
├── __init__.py
├── utils/
│   ├── __init__.py
│   ├── hardness.py       # H_graph, H_GF, classical hardness calculators
│   ├── runners.py        # Seeded multi-run harness
│   ├── instances.py      # Standardized test instance generators
│   └── plotting.py       # Shared plotting utilities
├── exp1_delta_scaling.py
├── exp2_density_sweep.py
├── exp3_smoothness_asymptotics.py
├── exp4_kernel_comparison.py
├── exp5_competitive_set.py
└── exp6_fig1_with_ci.py
```

### 0.2 `utils/hardness.py` — Hardness measure calculators

Implement the following functions; all take `(means, Adj, Degree)` or equivalent:

- `classical_hardness(means)` → returns `sum_i 1/Delta_i^2` over suboptimal arms. This is the no-graph hardness.
- `graph_hardness(means, Adj, Degree, rho)` → returns `H_graph` from Theorem 4.1.
  - Identify the best arm `a_star`.
  - For each suboptimal arm `i`, compute the influence factor `J(i,G)` via resistance distance on the graph Laplacian (use `np.linalg.pinv` on `L_G` to get the effective-resistance matrix; `R_eff(i,j) = (L_G^+)_{ii} + (L_G^+)_{jj} - 2 (L_G^+)_{ij}`).
  - An arm is **competitive** if `rho * J(i,G)` is not large enough to dominate `1/Delta_{i,c}^2`; **non-competitive** otherwise. Use the threshold given in §4.1/Theorem C.1 of Thaker et al. 2022 (cited as `\cite{thaker2022maximizing}`).
  - Return `H_graph = sum over competitive arms of 1/Delta_{i,c}^2 + max over non-competitive arms of 1/Delta_{i,c}^2`.
- `graph_feedback_hardness(means, Adj)` → returns `H_GF` from eq. (7) by solving the covering LP:
  ```
  minimize  sum_a tau_a
  subject to  sum_{a : i in N+(a)} tau_a >= 1/Delta_i^2   for all i != i*
              tau_a >= 0
  ```
  - Use `scipy.optimize.linprog` with `method='highs'`.
  - The constraint matrix `A_ub` has rows indexed by suboptimal arms and columns by actions; entry is `-1` if `i ∈ N+(a)` else `0` (inequality is flipped to `<=`).
  - RHS is `-1/Delta_i^2`.
- `rho_star(epsilon, K, T_estimate, delta, sigma=1.0)` → returns the optimal regularization `rho^*(epsilon) = sigma_0 * sqrt(L_1(T)) / epsilon` from eq. (5) of the paper, where `sigma_0 = 2 * sigma * sqrt(14)` and `L_1(T) = log(12 K^2 T^2 / delta)`. Since this depends on the (unknown) stopping time `T`, take `T_estimate` as input — in practice, iterate: start with a rough upper bound on `T` (e.g., `classical_hardness * log(1/delta)`), tune `rho^*` from that, run the algorithm, and if the actual stopping time differs substantially re-tune and re-run (or simply report results for a plausible range of `T_estimate`).
- `epsilon_hardness(means, Adj, Degree, epsilon, T_estimate, delta, sigma=1.0)` → returns `H_epsilon` from Theorem 4.4, computed using the competitive set from Definition 4.3:
  - Compute `rho = rho_star(epsilon, ...)`.
  - For each suboptimal arm `i`, classify it as competitive iff `Delta_{i,c}^2 * J(i,G) <= c_0 * sigma_0 * epsilon * sqrt(L_1(T))` with `c_0 = 4` (from the proof in Appendix B of the paper).
  - Return `H_epsilon = sum over competitive arms of 1/Delta_{i,c}^2 + max over non-competitive arms of 1/Delta_{i,c}^2`.
- `critical_epsilons(means, Adj, Degree, T_estimate, delta, sigma=1.0)` → returns a dict `{i: eps_i_star}` of the critical smoothness levels from eq. (6) of the paper, for each suboptimal arm `i`. This is what lets Experiment 3 predict phase transitions.

Also provide `competitive_set(means, Adj, Degree, rho)` → returns `(H_competitive_indices, N_noncompetitive_indices)` for the fixed-$\rho$ setting (Experiment 5 needs this), and `competitive_set_epsilon(means, Adj, Degree, epsilon, T_estimate, delta, sigma=1.0)` for the $\varepsilon$-indexed version (Experiment 3 needs this).

### 0.3 `utils/runners.py` — Seeded multi-run harness

Provide a function:

```python
def run_algorithm(algo_factory, seed, max_steps=1_000_000):
    """
    Run a single algorithm with a given seed until stopping condition.

    Parameters
    ----------
    algo_factory : callable that takes no args and returns an algorithm instance.
                   Must be called AFTER np.random.seed(seed) is set.
    seed : int, numpy random seed.
    max_steps : int, hard cap to prevent runaway experiments.

    Returns
    -------
    dict with keys:
        'stopping_time' : int, total number of real arm pulls when algorithm terminated
        'selected_arm'  : int, arm the algorithm returned
        'correct'       : bool, whether selected_arm == argmax(true_means)
        'elimination_curve' : list[tuple[int, int]], (t, num_remaining) for plotting
        'pull_counts'   : np.ndarray, direct pulls per arm at termination
        'converged_flag': bool, whether the algo hit a proper stopping rule
    """
```

Also provide:

```python
def run_many(algo_factory, seeds, n_jobs=1):
    """Run in parallel (multiprocessing) across seeds, return list of results."""
```

**Important:** The "stopping time" must be the number of *real* arm pulls, not the number of TS rounds. Each TS round pulls one real arm regardless of how many Thompson samples `M(delta, q, t)` are drawn. The existing code tracks this correctly via `self.t` (incremented once per call to `play_arm`), but verify this when wiring up the runner.

### 0.4 `utils/instances.py` — Standardized test instances

Wrap `graph_generator.call_generator` with named instance builders:

```python
def sbm_standard(n_clusters=10, nodes_per_cluster=10, p=0.9, q=0.0,
                 best_factor=1.3, seed=0):
    """Standard SBM with one isolated best arm. Returns (means, Adj, Degree)."""

def erdos_renyi(n=50, p=0.2, gap=0.3, seed=0):
    """ER graph with means on two clusters (best cluster rewards = 1, rest = 1-gap)."""

def barabasi_albert(n=100, m=2, gap=0.3, seed=0):
    """BA scale-free graph for heterogeneous-degree experiments."""

def complete_graph(n=50, gap=0.3, seed=0):
    """Clique."""

def empty_graph(n=50, gap=0.3, seed=0):
    """No edges (classical bandit baseline)."""
```

Each returns `(means, Adj, Degree)` as NumPy arrays. The best arm should always be at index 0.

### 0.5 `utils/plotting.py` — Shared plotting helpers

Provide:

- `plot_with_ci(ax, x, runs, label, color)` — plots median with 25–75 percentile shading across seeds.
- `nice_log_axis(ax, which='x')` — formats log-scale axes cleanly.
- Consistent color/marker dictionary per algorithm name.

### 0.6 Performance fix to `ThompsonSampling.get_all_teff`

The current implementation calls `np.linalg.inv(M)` every round, giving O(n³) per-round cost. Replace with:

- Maintain `self.inverse_tracker` as currently done in `algobase.AlgoBase.play_arm`, which already uses Sherman–Morrison.
- Replace `get_all_teff` with:
  ```python
  def get_all_teff(self):
      return 1.0 / np.diag(self.inverse_tracker)
  ```

Verify on a small instance (n=20) that this gives numerically identical results to the old version, then remove the old one. This fix is required to make Experiment 1 (δ-sweep) tractable.

### 0.7 Add `GraphFeedbackTS` class to `graph_algo.py`

Implement Algorithm 2 from §4.3 of the paper (the graph-feedback section; in the revised version, §4.2 is now the misspecification section, and graph feedback moved to §4.3). Key differences from `ThompsonSampling`:

- Estimator is the empirical mean of **feedback observations**, not Laplacian-regularized.
- Maintain `N_fb[i]` = number of times arm `i` has been observed through any pull's closed neighborhood.
- Maintain `R_fb[i]` = sum of rewards observed for arm `i`.
- `mu_hat[i] = R_fb[i] / N_fb[i]`.
- Variance for TS sample on arm `i` is `C(t) / N_fb[i]`.
- **Pulling rule** (critical — different from graph-smooth):
  - Upon disagreement, identify pair `(i_hat, i_tilde)`.
  - Among all actions `a`, choose one that maximizes `|N+(a) ∩ {i_hat, i_tilde}|` (the neighborhood that covers the most candidates).
  - Break ties by the smaller feedback count `N_fb[a]`.
- **Initialization**: pull a covering set of arms (a dominating set of `G`) so every arm has `N_fb[i] >= 1`. A greedy dominating-set construction is fine.
- **Reward observation**: when arm `a` is pulled, for **every** `j ∈ N+(a)`, sample a reward `r_j ~ N(mu[j], sigma^2)` and update `N_fb[j] += 1`, `R_fb[j] += r_j`.

Test the class on a simple 5-node clique first — verify that one pull updates all 5 arms.

### 0.8 Expose $\rho$ and $\varepsilon$ as independent knobs in `ThompsonSampling`

Experiment 3 needs to sweep $\varepsilon$ while simultaneously setting $\rho = \rho^\star(\varepsilon)$. The current `AlgoBase.__init__` and `initialize_conf_width` do two things that interfere with this:

1. `AlgoBase.L_rho = self.eta * self.L + self.rho * np.identity(self.dim)` — this mixes the Laplacian weight (`eta`, which corresponds to the paper's $\rho$) with a small diagonal regularizer (`self.rho`, currently hardcoded to `1e-4`). Confusingly, the code's `self.rho` is *not* the paper's $\rho$.

2. `initialize_conf_width` overwrites `self.eps` by computing the true smoothness from the provided means (line 80). This makes it impossible to run the algorithm with a nominal $\varepsilon$ different from the true one.

Required changes:

- Rename the code's `self.eta` → `self.rho_lap` (the paper's $\rho$, Laplacian weight) and keep the existing `self.rho` as `self.rho_diag` (small diagonal regularizer for invertibility). Update all references.
- Add a constructor argument `epsilon_nominal` that, if provided, overrides the auto-computed smoothness. Default behavior (auto-compute from true means) is preserved when `epsilon_nominal=None`.
- Expose `rho_lap` and `epsilon_nominal` as constructor arguments of `ThompsonSampling` so that Experiment 3 can pass them explicitly.
- Update the confidence-width formula in `eliminate_arms` (line 146 of `algobase.py`) to use `self.rho_lap * self.epsilon_nominal` as the bias term, consistent with the paper's $\rho\varepsilon$ product.

This is a mechanical refactor but touches several files. Do it once, carefully, then verify with a regression test: on the default `config.toml` instance, `ThompsonSampling` should produce identical stopping times to the pre-refactor version when called with `epsilon_nominal=None` and `rho_lap=1.0`.

### 0.9 Checkpoint: human review before running experiments

**Stop here for human review.** Phase 0 produces the shared infrastructure on which every experiment depends. In particular, `utils/hardness.py` computes the ground-truth quantities that every acceptance criterion compares against — if $H_{\mathrm{graph}}$, $H_{\mathrm{GF}}$, $H_\varepsilon$, or $\mathcal{J}(i,G)$ is miscomputed, every downstream plot will be wrong in a way that's hard to detect from the plot alone.

**Before starting Phase 1, the agent should produce a short verification report** (`experiments/outputs/phase0_verification.md`) containing:

1. **Hardness calculators on small instances with closed-form answers.**
   - 3-arm empty graph (no edges), $\Delta_1 = 0.2$, $\Delta_2 = 0.4$. Expected `classical_hardness = 1/0.04 + 1/0.16 = 31.25`. Expected `graph_hardness` ≈ same (no edges means no graph benefit). Expected `H_GF` ≈ same (covering LP with disjoint constraints is just the sum).
   - 3-arm clique, same gaps. Expected `H_GF = max(1/0.04, 1/0.16) = 25` (clique LP solved by pulling the weakest-constrained arm).
   - 2-cluster SBM with within-cluster means identical, one isolated best arm. Verify `J(i,G)` is finite and positive for every arm; verify `graph_hardness < classical_hardness`.
2. **Regression test output for the refactored `ThompsonSampling`.** Run the pre-refactor and post-refactor versions on the `config.toml` instance with a fixed seed. Confirm identical stopping times (to within seed-level reproducibility — exact equality if the refactor was purely renames, small variance if it changed initialization order).
3. **Performance measurement for the Sherman–Morrison fix.** Time one `play_round` call on $n=100, 200, 400$ arms, before and after the fix. Confirm the new code is at least 5× faster at $n=400$.
4. **`GraphFeedbackTS` unit tests:**
   - 5-node clique, pull arm 0: confirm all 5 arms have $N_{\mathrm{fb}}^i = 1$ afterwards.
   - 2-node empty graph: confirm pulling arm 0 updates only arm 0.
5. **`rho_star()` smoke test.** For $\varepsilon = 1, 0.1, 0.01$, print $\rho^\star$. Check that the values are on the order of $\sqrt{L_1(T)}/\varepsilon$ by hand.

If any check fails, fix it before proceeding. The experiments are designed to produce scientifically meaningful signals only if the infrastructure is correct.

---

## Phase 1: Experiment 1 — δ-scaling on a fixed instance

**Validates:** Theorem 4.1 — sample complexity scales as `H_graph * log(1/delta) + H_graph * log^2(K * H_graph)`.

**File:** `experiments/exp1_delta_scaling.py`

### Setup

- Instance: `sbm_standard(n_clusters=10, nodes_per_cluster=10, p=0.9, q=0.0, best_factor=1.3)` → 101 arms.
- `delta` sweep: `[1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]`.
- Algorithms:
  1. `ThompsonSampling` (graph-smooth TS with graph)
  2. `BasicThompsonSampling` (no-graph TS baseline)
  3. *(Optional)* `MaxDiffVarAlgo` as an additional UCB-based graph baseline.
- Seeds: 20 per `(algorithm, delta)` cell. Use seeds `range(20)`.
- Hyperparameters: `q = 0.01`, `eta = 1.0`, `rho = 1e-4` (matches paper and existing code).

### Procedure

1. For each `delta`, for each algorithm, for each seed:
   - Set `np.random.seed(seed)`.
   - Instantiate algorithm and run to convergence.
   - Record `stopping_time` and `correct`.
2. Assert that `correct` is `True` in at least `1 - delta` fraction of runs for each cell (sanity check on the theoretical guarantee).

### Output

- **Main plot:** x-axis = `log(1/delta)`, y-axis = stopping time. One line per algorithm. Shading = 25–75 percentile across seeds. Overlay two dashed horizontal reference lines: analytically computed `classical_hardness * log(1/delta)` and `graph_hardness * log(1/delta)` at one chosen delta.
- **Secondary plot:** log-log plot of stopping time vs `1/delta` to visually confirm linearity on the log-log transformation.
- **Table:** fit a linear regression `T = a * log(1/delta) + b` for each algorithm, report slope `a` and compare against theoretical `H_graph`.
- Save plot as `experiments/outputs/exp1_delta_scaling.png` at 150 dpi.
- Save raw results as `experiments/outputs/exp1_results.npz` with all seed-level data.

### Acceptance criteria

- `ThompsonSampling` slope is smaller than `BasicThompsonSampling` slope (graph helps).
- Fitted slope of `ThompsonSampling` is within 2× of theoretical `H_graph`.
- All runs return the optimal arm (error rate ≤ `delta`).

---

## Phase 2: Experiment 2 — Density sweep (graph-smooth vs graph-feedback)

**Validates:** Theorem 4.3 — `H_GF` is much smaller than `H_graph` on dense graphs and they converge on sparse graphs.

**File:** `experiments/exp2_density_sweep.py`

**Depends on:** `GraphFeedbackTS` class from Phase 0.7.

### Setup

- Instance: Erdős–Rényi graphs on `n = 50` nodes, best arm at index 0 with `mu[0] = 1.0`, all others `mu[i] = 0.7` (so `Delta_i = 0.3` for all suboptimals — simplifies hardness calculations).
- `p` sweep: `[0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]`.
- Algorithms:
  1. `ThompsonSampling` (graph-smooth)
  2. `GraphFeedbackTS` (graph feedback)
  3. `BasicThompsonSampling` (no-graph baseline)
- Seeds: 15 per `(algorithm, p)` cell.
- `delta = 1e-3`.

### Procedure

1. For each `p`, generate **one fixed ER graph** (seed the graph construction independently so all algorithms see the same graph at each `p`).
2. For each algorithm, for each seed, run once with that seed.
3. Compute `H_graph`, `H_GF`, classical hardness analytically for each `p`; record.

### Output

- **Main plot:** x-axis = edge probability `p`, y-axis = median stopping time (log scale). Three lines with 25–75 shading.
- **Overlay:** three dashed reference lines — `H_graph(p)`, `H_GF(p)`, classical hardness — scaled by `log(1/delta)` constant.
- **Expected pattern:**
  - At `p=1` (clique): `GraphFeedbackTS` ≈ `H_GF = max_i 1/Delta_i^2` → smallest; `ThompsonSampling` still benefits but less; `BasicThompsonSampling` worst.
  - At `p=0.05` (very sparse): all three converge.
- Save to `experiments/outputs/exp2_density_sweep.png`.

### Acceptance criteria

- `GraphFeedbackTS` stopping time at `p=1.0` is at least 5× smaller than `ThompsonSampling`.
- At `p=0.05`, the three algorithms are within 2× of each other.
- Trends are monotone in `p` (allow noise, but medians should move in the expected direction).

---

## Phase 3: Experiment 3 — Smoothness asymptotics and phase transitions

**Validates:** Theorem 4.4 (`thm:main-mis`) and Corollary 4.5 (`cor:eps-limit`) — as $\varepsilon \to 0$ with $\rho = \rho^\star(\varepsilon)$, the competitive set shrinks through a sequence of phase transitions, and sample complexity approaches $\max_{i \neq a^\star} \Delta_{i,c}^{-2} \cdot \mathrm{polylog}(K, 1/\delta)$.

**File:** `experiments/exp3_smoothness_asymptotics.py`

**Important:** This is a **different** experiment from what a graceful-degradation test would look like. We are not corrupting the means. Instead, we fix the instance (and hence the true $\|\mu\|_G$), sweep the nominal smoothness parameter $\varepsilon$ downward, re-tune $\rho$ to $\rho^\star(\varepsilon)$ at each value, and observe (a) the stopping time approaching a limit determined by $\max_i \Delta_{i,c}^{-2}$ rather than the sum, and (b) discrete drops in the stopping time as individual arms cross their critical $\varepsilon_i^\star$.

### Pre-sweep numerical sanity check

Before running the full $\varepsilon$ sweep, verify the algorithm is numerically well-behaved across the intended range. The issue: $\rho^\star(\varepsilon) = \sigma_0\sqrt{L_1(T)}/\varepsilon$ scales as $1/\varepsilon$, so at $\varepsilon = 10^{-3}$ with typical $T$ and $\delta$ we get $\rho^\star \sim 10^4$. This is four to five orders of magnitude larger than the `rho = 0.0001` diagonal regularizer hardcoded in the current `AlgoBase`, and may cause:

- **Matrix conditioning:** $V_t = \sum e_{\pi_s}e_{\pi_s}^\top + \rho K_G$ becomes dominated by $\rho K_G$. Since $L_G$ has a zero eigenvalue (the all-ones vector), the `rho_diag` term (paper's small-$I$ regularizer) is the only thing keeping $V_t$ invertible in the null direction. If `rho_diag` is not scaled up proportionally, the condition number blows up.
- **Effective sample size overflow:** $t_{\mathrm{eff},i}(t) = 1/[V_t^{-1}]_{ii}$ can become astronomically large at small $\varepsilon$, which may interact badly with the $C(t)/t_{\mathrm{eff},i}$ variance in Thompson samples.
- **Bias swamping:** the term $\rho^\star \varepsilon$ in the confidence radius should *equal* $\sigma_0\sqrt{L_1(T)}$ by construction (that's the point of $\rho^\star$), but if `epsilon_nominal` and `rho_lap` are passed inconsistently, this balance breaks and the bias term can dominate.

**Implement a `probe_rho_star()` function** as a standalone script before the full sweep. For each $\varepsilon$ in the planned sweep:

1. Compute $\rho^\star(\varepsilon)$ and print it.
2. Form $V_t = \rho^\star L_G + \mathtt{rho\_diag} \cdot I$ (at $t=0$, before any pulls). Report `np.linalg.cond(V_t)`.
3. Compute $t_{\mathrm{eff},i}(0) = 1/[V_t^{-1}]_{ii}$ for every arm; report min, max, and median.
4. Compute the nominal bias term $\rho^\star \cdot \varepsilon$ — this should equal $\sigma_0\sqrt{L_1(T)}$ regardless of $\varepsilon$. If the two numbers disagree by more than 1%, there's a bug in `rho_star()`.

**Acceptance for the sanity check:**
- $\mathrm{cond}(V_t) < 10^{10}$ for every $\varepsilon$ in the sweep.
- $t_{\mathrm{eff},i}(0)$ values are finite and positive for every arm.
- $\rho^\star \varepsilon \approx \sigma_0\sqrt{L_1(T)}$ at 1% tolerance for every $\varepsilon$.

**If the check fails:**
- **High condition number:** scale `rho_diag` proportionally to $\rho^\star$ — e.g., `rho_diag = max(1e-4, 1e-6 * rho_lap)`. This preserves invertibility without materially altering the Laplacian-dominant regime.
- **Effective sample size overflow:** if $t_{\mathrm{eff},i}(0) > 10^{12}$ at small $\varepsilon$, restrict the $\varepsilon$ sweep to a smaller range (e.g., $\varepsilon \in [10^{-2}, 10^0]$) and update the Setup section accordingly. You'll lose some of the asymptotic-limit evidence but still see phase transitions.
- **Bias mismatch:** re-audit the code path from `epsilon_nominal` → confidence radius. The product $\rho \cdot \varepsilon$ should appear exactly once, multiplicatively.

Only proceed to the full sweep once all three checks pass. Save the probe output to `experiments/outputs/exp3_sanity_check.txt` as an audit trail.

### Setup

- **Instance:** A designed instance with a clear hierarchy of critical smoothness levels. Use an SBM with 5 clusters × 6 nodes + 1 isolated best arm (31 arms). Cluster means chosen so that $\Delta_{i,c}^2 \mathcal{J}(i,G)$ takes 5 well-separated values across clusters (so we get 5 distinct phase transitions as $\varepsilon$ decreases).
  - Concretely: pick cluster means $\{\mu_1, \mu_2, \mu_3, \mu_4, \mu_5\} = \{0.9, 0.7, 0.5, 0.3, 0.1\}$ and best arm $\mu_0 = 1.0$. Verify that the resulting $\varepsilon_i^\star$ values span at least 2 orders of magnitude.
  - Critical: the algorithm must use the **true** $\|\mu\|_G$ when forming the concentration radius, but the algorithm designer gets to choose the nominal $\varepsilon$ and the corresponding $\rho$. We are studying the effect of the designer's choice of $\varepsilon$ (with $\rho = \rho^\star(\varepsilon)$), not a mismatch between nominal and true smoothness.
- **$\varepsilon$ sweep:** logarithmically spaced, $\varepsilon \in \{10^0, 10^{-0.5}, 10^{-1}, \ldots, 10^{-3}\}$ (7 points). The lower end should be well below the smallest $\varepsilon_i^\star$ in the instance.
- **$\rho$ choice:** for each $\varepsilon$, set $\rho = \rho^\star(\varepsilon)$ using `rho_star()` from Phase 0.2. Use `T_estimate = classical_hardness * log(1/delta)` for the initial estimate.
- **Algorithms:**
  1. `ThompsonSampling` with $\rho = \rho^\star(\varepsilon)$, nominal smoothness = $\varepsilon$. This is the "optimally-tuned" variant.
  2. `ThompsonSampling` with $\rho = 1$ fixed (no re-tuning). This isolates the benefit of $\rho$-tuning vs. just using a small $\varepsilon$.
  3. `BasicThompsonSampling` (no-graph baseline, $\varepsilon$-independent — flat reference line).
- **Seeds:** 20 per $(\text{algorithm}, \varepsilon)$ cell.
- **$\delta = 10^{-3}$.**

### Procedure

1. Compute $\varepsilon_i^\star$ for every suboptimal arm $i$ using `critical_epsilons()`. Record the values — they predict where phase transitions will occur in the plot.
2. For each $\varepsilon$ in the sweep:
   - Compute $\rho^\star(\varepsilon)$.
   - Compute $H_\varepsilon$ analytically using `epsilon_hardness()`.
   - Compute the competitive set under this $\varepsilon$.
3. For each seed, run each algorithm. Record stopping time, correctness, and per-arm pull counts.

### Output

Three-panel figure:

- **Panel A (phase transition plot):** x-axis = $\log_{10}(\varepsilon)$ (decreasing left-to-right, so phase transitions happen as you move right), y-axis = median stopping time (log scale).
  - Three lines: optimally-tuned TS, $\rho=1$ TS, and classical baseline.
  - Overlay vertical dashed lines at each $\log_{10}(\varepsilon_i^\star)$, labeled with arm index or gap magnitude. These are the predicted phase-transition locations.
  - Overlay a dashed horizontal line at the asymptotic limit $\max_{i \neq a^\star} \Delta_{i,c}^{-2} \cdot \log(1/\delta)$ (approximate polylog constant).
- **Panel B (hardness curve):** x-axis = $\log_{10}(\varepsilon)$, y-axis = $H_\varepsilon$ (analytical, from `epsilon_hardness`). This should be a staircase function with steps at each $\varepsilon_i^\star$. Overlay the two horizontal limits: $\sum_{i \neq a^\star} \Delta_{i,c}^{-2}$ (as $\varepsilon \to \infty$) and $\max_{i \neq a^\star} \Delta_{i,c}^{-2}$ (as $\varepsilon \to 0$).
- **Panel C (competitive set size):** x-axis = $\log_{10}(\varepsilon)$, y-axis = $|\mathcal{H}_\varepsilon|$. Step function decreasing from $K-1$ to $0$ as $\varepsilon$ decreases.
- Save to `experiments/outputs/exp3_smoothness_asymptotics.png`.

### Acceptance criteria

- **Monotonicity:** the optimally-tuned TS stopping time is monotone (weakly) decreasing in $\varepsilon$.
- **Asymptotic limit:** at the smallest $\varepsilon$, optimally-tuned TS stopping time is within 3× of the theoretical limit $\max_i \Delta_{i,c}^{-2} \log(1/\delta)$ (allowing for the polylog slack).
- **Phase-transition visibility:** at least 2 of the predicted $\varepsilon_i^\star$ locations in Panel A coincide (within a half-decade in $\varepsilon$) with a visible drop in median stopping time.
- **$\rho$-tuning matters:** at the smallest $\varepsilon$, optimally-tuned TS is at least 2× faster than $\rho=1$ TS. (If not, the benefit comes from $\varepsilon$ alone and the theory's re-tuning story is under-supported.)
- **Classical ceiling:** classical baseline stopping time is roughly constant across the $\varepsilon$ sweep (it shouldn't depend on $\varepsilon$).
- Error rate stays below $\delta$ throughout.

### Implementation notes

- The paper's Remark after the misspecification proof (Appendix B) clarifies that if the algorithm's nominal $\varepsilon$ differs from the true $\|\mu\|_G$, the analysis still goes through as long as $\rho\varepsilon \geq \rho \|\mu\|_G$. The algorithm's confidence radius uses `eta * self.eps` (in current code) as a proxy for $\rho\varepsilon$. Make sure the code correctly passes the **nominal** $\varepsilon$ (the value being swept) to the algorithm, not the true smoothness.
- If the existing `AlgoBase.initialize_conf_width` automatically overwrites `self.eps` by computing it from the true means (see `compute_imperfect_info` at line 80 of `algobase.py`), this needs to be modified to accept an externally supplied $\varepsilon$ that overrides the computed value.
- `rho_star` depends on $T$, which depends on $\rho$. For simplicity, fix `T_estimate = 10 * classical_hardness` in the initial implementation; the log dependence is mild. If results look off, iterate once by re-running with `T_estimate = actual_stopping_time` and checking for convergence.

---

## Phase 4: Experiment 4 — Laplacian vs normalized Laplacian

**Validates:** Theorem 4.4 — normalized Laplacian helps on heterogeneous-degree graphs.

**File:** `experiments/exp4_kernel_comparison.py`

### Setup

- Instances:
  1. Barabási–Albert graph: `n=100, m=2` (heavy-tailed degrees).
  2. SBM (for contrast — near-uniform degrees).
- Means: cluster-structured, best arm at index 0 with mean `1.0`, others at `0.7`.
- Algorithms: `ThompsonSampling` with two variants
  - `kernel='combinatorial'`: uses `L_G = D - A` (current default)
  - `kernel='normalized'`: uses `K_G = I - D^{-1/2} A D^{-1/2}` (handle isolated nodes carefully — if `deg(i) = 0`, set the corresponding row/col of `D^{-1/2}` to 0 and treat the arm as disconnected).
- Seeds: 20 per `(instance, kernel)` cell.

### Implementation note

Add a `kernel` parameter to `algobase.AlgoBase.__init__`:

```python
def __init__(self, D, A, mu, eta, add_graph=True, eps=0.0, kernel='combinatorial'):
    ...
    if kernel == 'combinatorial':
        self.L = D - A
    elif kernel == 'normalized':
        d_inv_sqrt = np.where(np.diag(D) > 0, 1.0/np.sqrt(np.diag(D)), 0.0)
        D_inv_sqrt = np.diag(d_inv_sqrt)
        self.L = np.eye(self.dim) - D_inv_sqrt @ A @ D_inv_sqrt
    ...
```

Everything else stays the same.

### Output

- **Bar chart:** two groups (BA, SBM), each with two bars (combinatorial, normalized). Bar = median stopping time; error bar = IQR.
- **Per-arm pull distribution:** for the BA instance, side-by-side histogram of pull counts across arms, one per kernel. Normalized version should show more balance (less weight on hub nodes).
- Save to `experiments/outputs/exp4_kernel_comparison.png`.

### Acceptance criteria

- On BA, normalized kernel median stopping time ≤ combinatorial (expect 10–30% improvement).
- On SBM, the two kernels give similar results (within 10%).
- Pull count variance is lower for normalized on the BA graph.

---

## Phase 5: Experiment 5 — Competitive vs non-competitive decomposition

**Validates:** the mechanistic structure of `H_graph` — non-competitive arms cost `O(1)` direct pulls.

**File:** `experiments/exp5_competitive_set.py`

**Depends on:** `competitive_set()` in `utils/hardness.py`.

### Setup

- Instance: `sbm_standard` with 10 clusters × 10 nodes + 1 isolated best arm. Cluster means linearly spaced. This creates a clear separation: some suboptimal clusters will have small gaps (competitive) and some large gaps (non-competitive).
- Pre-compute `H_set` and `N_set` analytically.
- Run `ThompsonSampling` for 30 seeds with `delta = 1e-4`.

### Procedure

1. Pre-compute competitive/non-competitive partition using `competitive_set(means, Adj, Degree, rho)`.
2. For each seed, run TS, record `pull_counts` at termination.
3. Aggregate: for each arm, compute median pull count across seeds.

### Output

Four-panel figure:

- **Panel A:** Bar chart of median pull count per arm, colored by membership (competitive = blue, non-competitive = orange, best = green). Arms sorted by true gap.
- **Panel B:** Scatter plot of `1/Delta_{i,c}^2` (x-axis) vs median pull count (y-axis) for competitive arms only. Should show a roughly linear relationship.
- **Panel C:** Histogram of pull counts for non-competitive arms. Should concentrate near the minimum.
- **Panel D:** Bar chart of `pull_count * Delta_{i,c}^2` per arm — roughly constant on competitive set, near zero on non-competitive.
- Save to `experiments/outputs/exp5_competitive_set.png`.

### Acceptance criteria

- Non-competitive arms receive O(log T) pulls on median (not O(1/Delta^2)).
- Competitive arm pull counts scale roughly as `1/Delta_{i,c}^2` (Pearson correlation > 0.7 in Panel B).

---

## Phase 6: Experiment 6 — Figure 1 with seed variance

**Purpose:** Replace the misleading single-run Figure 1 in the paper with a version that shows variance properly.

**File:** `experiments/exp6_fig1_with_ci.py`

### Setup

- Same instance as current `sample_main.py`: reads from `config.toml` (101 arms, SBM).
- Algorithms: all of {`MaxDiffVarAlgo` (MVM), `CyclicAlgo`, `OneStepMinSumAlgo` (JVM-N), `MaxVarianceArmAlgo` (JVM-O), `ThompsonSampling`, `NoGraphAlgo`}.
- Seeds: 30.

### Procedure

1. For each seed, each algorithm, record the full elimination curve `[(t, num_remaining)]`.
2. For UCB methods: interpolate each curve onto a common time grid, then take 25/50/75 percentiles.
3. For `ThompsonSampling`: it doesn't produce intermediate eliminations. Record only the stopping time. Represent it in the plot as a horizontal violin plot or a shaded vertical band `[t_25, t_75]` at the bottom of the figure.

### Output

- Main plot: same style as existing Figure 1 but with median curves and shaded IQRs for UCB methods, plus a vertical shaded band (or violin on a secondary axis) for TS stopping times.
- Save to `experiments/outputs/exp6_fig1_with_ci.png`.

### Acceptance criteria

- TS stopping time band is strictly left of all UCB method medians (with no overlap).
- UCB methods show non-trivial IQR (confirms that seed variance matters).

---

## Running the full experimental suite

Create a top-level script `experiments/run_all.py` that:

1. Creates `experiments/outputs/` if it doesn't exist.
2. Runs all six experiments in sequence.
3. Uses `multiprocessing.Pool` to parallelize seeds within each experiment (use all available cores minus 1).
4. Prints a summary table at the end: for each experiment, whether acceptance criteria passed.

**Expected total runtime:** On a modern 8-core laptop, with the Sherman–Morrison fix in place, the full suite should complete in 2–4 hours. Experiment 1 (δ-sweep) is the longest due to `delta=1e-6` runs.

---

## Deliverables for the paper

After all experiments pass:

- `experiments/outputs/exp1_delta_scaling.png` → goes in §5 as the main theoretical-validation figure for Theorem 4.1.
- `experiments/outputs/exp2_density_sweep.png` → supports the graph-feedback section §4.3 and its comparison paragraph.
- `experiments/outputs/exp3_smoothness_asymptotics.png` → supports §4.2 (optimal tuning and asymptotics) and directly visualizes Corollary 4.5.
- `experiments/outputs/exp4_kernel_comparison.png` → supports §4.4 (PSD kernel generalization) motivation.
- `experiments/outputs/exp5_competitive_set.png` → goes in the appendix as mechanistic validation of the $H_{\mathrm{graph}}$ structure.
- `experiments/outputs/exp6_fig1_with_ci.png` → replaces current Figure 1.

Figure 2 from `sample_main2.py` (stopping time vs number of arms) can be kept as-is but should be re-run with seed averaging using the new `runners.py` infrastructure — this is a small extension of Experiment 1 and can be added to that script.

---

## Open questions / decisions for the researcher

1. **Which instance should Experiment 1 use?** The SBM in `config.toml` is reasonable but the gap structure (`best_factor=1.3`) gives only one "hard" suboptimal cluster. Consider whether to also run on a harder instance with smaller gaps (e.g. `best_factor=1.05`) to stress-test.

2. **Influence factor computation.** The GRUB paper (Thaker et al. 2022) defines `J(i,G)` via resistance distance. The `utils/hardness.py` implementation should cite the exact formula used (add a docstring reference). If there's ambiguity, default to the formulation in Theorem C.1 of that paper.

3. **Covariance matrix estimator in `GraphFeedbackTS`.** The paper uses empirical means (Hoeffding-based concentration). If numerical stability becomes an issue (e.g., arms with very small `N_fb`), add Bayesian prior `N_fb_i <- N_fb_i + 1` uniformly.

4. **Instance design for Experiment 3.** The phase-transition plot only works if the $\varepsilon_i^\star$ values are well separated. If the initial instance gives clustered $\varepsilon_i^\star$ values (e.g., all within a factor of 2), tune the cluster means or influence factors to spread them out. Alternatively, if phase transitions are too subtle to see in Panel A, fall back to just showing the limit behavior: that at small $\varepsilon$, TS stopping time matches $\max_i \Delta_{i,c}^{-2}$ (the asymptotic limit from Corollary 4.5), even if the intermediate staircase is noisy.

5. **Value of $c_0$ in $\varepsilon_i^\star$.** The paper's Appendix B proof gives $c_0 = 4$ explicitly. Use this value, but also sweep $c_0 \in \{1, 4, 16\}$ in Experiment 3 to see how sensitive the phase transition locations are to this constant. If the phase transitions are visible at one $c_0$ but not others, report which.

6. **$T$-estimate bootstrapping in `rho_star`.** The optimal $\rho^\star$ depends on $\sqrt{L_1(T)}$, which is a slowly varying function of $T$. If results are sensitive to the initial $T$ estimate, implement a two-iteration scheme: run once with $T_{\text{est}} = \text{classical\_hardness}$, then re-run with $T_{\text{est}}$ set to the observed stopping time. Report both.

7. **Compute budget.** If 4 hours is too long, Experiment 1 can be compressed by dropping the `delta=1e-6` point (loses a little scaling-law evidence) and reducing seeds from 20 to 10. Don't compress Experiment 3 — the phase-transition plot needs variance to be credible, and the $\varepsilon$ sweep itself is the point of the experiment.