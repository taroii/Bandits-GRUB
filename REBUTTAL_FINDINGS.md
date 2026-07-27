# Rebuttal-round experiments: measurements and observations

Record of what was run and what was measured for the tasks in `paper/plan.md`.

**This document deliberately does not draw conclusions or make
recommendations.** It states what each source says, what the code does, and
what the runs produced. Interpretation is left to the reader.

Conventions: all values are medians over the stated seed count unless noted.
Any cell that reached the per-run pull budget is marked as a cap-hit and its
median is reported as a lower bound, not as a stopping time. No default
behaviour was changed; all new runs write to new `.npz` files and the committed
figure data was verified unchanged by md5.

> **Provenance of every number below.** These were produced on a laptop
> (macOS/arm64, numpy 2.0.2 + Apple Accelerate, 14 cores) at 3–5 seeds, and the
> sweeps were **stopped part-way** to move the work to a server. Cells not
> reached are marked *(not run)*. The raw `.npz` files are preserved under
> `experiments/outputs/mac_pilot/`, deliberately outside the resume path so a
> server run starts cold rather than mixing BLAS backends — the two differ by
> 0.1–0.6% on GRUB stopping times (§13). Re-run with `bash run_server.sh` for
> single-machine numbers at full seed counts; expect the values here to move by
> ~1% and the 5-seed medians to move more than that.

---

## 1. GRUB confidence radius (Task 0)

### What each source states

**Thaker et al. 2022** (arXiv 2108.01152**v2** LaTeX source,
`Content/Appendix_Resub_Arxiv/mean_variance.tex`, Lemma
`variance_estimate_proof`; the algorithm is named GRASP-UCB in v2). Verbatim:

```latex
|\hat{\mu}^{i}_T - \mu_{i}| \leq \sqrt{\frac{1}{t_{\text{eff}, i}}}
   \left(2\sigma\sqrt{14\log{\left(\frac{2w_i(\pmb{\pi}_T)}{\delta}\right)}}
   + \rho\|\pmb{\mu}\|_G\right)
```

with `w_i(π_T) = a_0 n t_eff,i²`. Their Algorithm pseudocode line
(`psuedocode.tex:271`): `β_i(t) ← 2σ√(14 log(2n t_eff,i²/δ)) + ρε`.

Their derivation's final step, verbatim from the same file:

```latex
|\langle \mathbf{e}_{i}, \rho V_T^{-1}L_G\pmb{\mu}\rangle|
  \leq \rho \sqrt{\langle \mathbf{e}_i ,V_T^{-1}\mathbf{e}_i\rangle}
       \sqrt{\langle L_G\pmb{\mu}, V_T^{-1}L_G\pmb{\mu}\rangle}
  \leq \rho\sqrt{[V_T^{-1}]_{ii}}\|\pmb{\mu}\|_G
```

The last inequality uses `⟨L_Gμ, V⁻¹L_Gμ⟩ ≤ ⟨μ, L_Gμ⟩`. Applying
`V_t ⪰ ρL_G ⇒ V⁻¹ ⪯ (1/ρ)L_G^†` on `range(L_G)` instead gives
`⟨L_Gμ, V⁻¹L_Gμ⟩ ≤ ε²/ρ`, hence `√ρ·ε` in place of `ρε`.

**Our Appendix H.1, line 1108:** "GRUB uses the radius
`β_i(t) = 2σ√(14 log(2Kt/δ)) + ρε` of [47]".

**What `algobase.py:eliminate_arms` did at HEAD:** computed
`beta_tracker = conc + 0.5·ρ·ε` and did not use it in the elimination. The
elimination used `conc_radius = 2√(14·log₂(2Kt/δ))` (base-2 logarithm) and
added `bias_vec = V⁻¹(0.5·ρ·L_G μ)` to the **lower bound only** (asymmetric,
signed, and using the true `μ`).

Numerically, `log₂` vs `ln` multiplies the noise term by 1.201 (a fixed
factor, verified across t). At ρ=100, ε=0.3 the three bias magnitudes are
`ρε = 30`, `√ρ·ε = 3`, and the exact oracle bias `≤ 0.27`, against a noise
term `2√(14 log(2Kt/δ))` of 38–48 over the range t ∈ [10³, 10⁷].

### Measurements

`grub_bias={legacy,published,sqrt,oracle,none}` was added to
`algobase.AlgoBase`. Default is `legacy`, which was verified **bit-exact**
against a pristine `git archive HEAD` checkout (`803738, 844702, 832562` on
chain K=10 seeds 0–2 from both).

GRUB medians, 5 seeds. Chain at ρ=100; MovieLens at K=20:

| bias | K=10 | K=20 | K=50 | K=100 | K=200 |
|---|---|---|---|---|---|
| `published` (ρε, ln) | 2.01e6 | 4.04e6 | cap 5/5 | cap 5/5 | *(not run)* |
| `legacy` (HEAD) | 8.45e5 | 1.77e6 | 4.63e6 | 9.27e6 | *(not run)* |
| `sqrt` (√ρ·ε, ln) | 6.79e5 | 1.42e6 | 3.68e6 | 7.19e6 | *(not run)* |
| `none` (no bias, ln) | 5.78e5 | 1.20e6 | 3.10e6 | 6.10e6 | *(not run)* |

For reference, the committed `main_2_results.npz` (20 seeds, `legacy`) has GRUB
at 8.42e5 / 1.77e6 / 4.56e6 / 9.32e6 / cap for K = 10 / 20 / 50 / 100 / 200.

| bias | ρ=1 | ρ=3 | ρ=10 | ρ=30 | ρ=100 | ρ=300 | ρ=1000 |
|---|---|---|---|---|---|---|---|
| `published` | 6.05e6 | 8.21e6 | cap 5/5 | cap 5/5 | cap 5/5 | cap 5/5 | *(not run)* |
| `legacy` | 7.32e6 | 7.35e6 | 7.36e6 | 7.44e6 | 7.34e6 | 7.44e6 | 7.23e6 |
| `sqrt` | 6.05e6 | 6.82e6 | 8.49e6 | cap 5/5 | cap 5/5 | cap 5/5 | cap 5/5 |
| `none` | 4.96e6 | 4.98e6 | 5.02e6 | 4.99e6 | 4.95e6 | 5.04e6 | 4.94e6 |

For reference, the committed `movielens_1_results.npz` (20 seeds, `legacy`) has
GRUB at 7.40e6 / 7.41e6 / 7.39e6 / 7.44e6 / 7.38e6 / 7.36e6 / 7.21e6 across the
same ρ grid.

Correctness was 100% in every cell above, including capped ones.

Arithmetic on the same numbers, stated without interpretation:

* Ratio of consecutive K cells (10→20→50): `published` 2.01, —;
  `legacy` 2.09, 2.62; `sqrt` 2.09, 2.59; `none` 2.07, 2.59.
* `legacy / published` at chain ρ=100, K=10: 0.42. At MovieLens ρ=1: 1.21.
* `none` vs TS-Explore (committed, 2.5e4) at chain K=10: 23.1×.
* Under `published`, the first capped chain cell is K=50; under `legacy` the
  committed data caps first at K=200. Appendix H.1 line 1119 states "The only
  configuration that hit this cap is GRUB at K = 200".
* On MovieLens, `none` varies by 1.9% across ρ ∈ [1, 1000] and `legacy` by
  2.9% across the same range (both with no cap-hits). `sqrt` is capped for
  ρ ≥ 30 and `published` for ρ ≥ 10.

Figure: `experiments/outputs/grub_bias.pdf` (cap-hits drawn as hollow markers).

### Two further H.1 / code differences observed

* H.1 line 1079 states `ρ_diag = max(1e-4, 1e-6ρ)`. That expression appears in
  `kernel_1.py`, `mis_2.py`, and `misc/mis_1.py`. It does not appear in
  `main_2.py` or `movielens_1.py`, which use the `AlgoBase` default `1e-4`. At
  ρ=1e3 the two expressions differ by 10×. The same value is used for
  TS-Explore and GRUB within any given runner.
* H.1 line 1080 states `V_t⁻¹` is computed by Cholesky factorization. The code
  uses `support_func.sherman_morrison_inverse` rank-one updates. Measured
  accuracy of the accumulated inverse against a fresh `np.linalg.inv` after
  50,000 rounds: max relative error 5.0e-10 (K=20, ρ=1), 4.3e-9 (K=20,
  ρ=1000), 3.5e-9 (K=200, ρ=100).

---

## 2. `hardness.rho_star` vs Eq. (5) (Task 3)

### What each source states

The paper states the prescription inline in three places (plain text, no
fraction rendering involved):

* main text line 47: "at any `ρ ≥ ρvar(ε) := σ₀² L₁(T)/ε²`"
* Appendix D line 673: "the bias floor ε once `ρ ≥ ρvar(ε) := σ₀² L₁(T)/ε²`"
* Appendix D line 702: "For any `ρ ≥ ρvar(ε) = σ₀² L₁/ε²`"

`hardness.rho_star` returns `σ₀√(L₁(T))/ε`. Verified numerically:
`rho_star(ε) == sqrt(σ₀²L₁/ε²)` exactly. The PAT review quotes Eq. (5) as
`σ₀√(L₁(T))/ε`.

On the connected SBM (K=31, σ₀=7.4833, T_est=4907, L₁=33.257):

| ε | `rho_star` | `σ₀²L₁/ε²` | ratio |
|---|---|---|---|
| 1e0 | 43.16 | 1,862 | 43.2 |
| 1e-1 | 431.6 | 1.862e5 | 431.6 |
| 1e-2 | 4,316 | 1.862e7 | 4,316 |
| 1e-3 | 4.316e4 | 1.862e9 | 4.316e4 |

Call sites of `rho_star`: `experiments/mis_2.py:165` (sets ρ for Appendix
H.3), `experiments/fig1_plot.py:51` and `experiments/movielens_1_plot.py:54`
(both compute `sigma0*sqrt(L1)/eps` inline for the dotted reference line),
and `misc/mis_1.py`.

`hardness.rho_var` was added returning `σ₀²L₁/ε²`. `rho_star` was left in
place with a docstring noting the relationship.

### Measurements

Committed `mis_2_results.npz` (ρ set by `rho_star`):

| ε | 1e0 | 3.16e-1 | 1e-1 | 3.16e-2 | 1e-2 | 3.16e-3 | 1e-3 |
|---|---|---|---|---|---|---|---|
| TS_tuned | 1,372,441 | 1,358,820 | 1,339,398 | 964,078 | 295,836 | 112,409 | 130,828 |

Basic TS on the same instance: 1,271,375. Minimum TS_tuned / Basic TS ratio:
11.31× at ε=3.16e-3.

Same instance with ρ set to `σ₀²L₁/ε_bar²` and fixed `ρ_diag=1e-4`
(`experiments/eps_sensitivity.py`, 5 seeds), ε_true = 0.4123:

| ε̄/ε_true | ε̄ | ρ | T | correct | cap |
|---|---|---|---|---|---|
| 0.125 | 0.0515 | 7.011e5 | 490,134 | 100% | 0/5 |
| 0.25 | 0.1031 | 1.753e5 | 236,134 | 100% | 0/5 |
| 0.5 | 0.2062 | 4.382e4 | 130,828 | 100% | 0/5 |
| 1.0 | 0.4123 | 1.096e4 | 116,303 | 100% | 0/5 |
| 2.0 | 0.8246 | 2,739 | 623,789 | 100% | 0/5 |
| 4.0, 8.0, 16.0 | | | *(not run)* | | |

Basic TS / T at ε̄/ε_true = 1: 1,271,375 / 116,303 = 10.93×.

MovieLens reference-line values (K=20, ε=3.408, T_est = median TS-Explore
median = 1,120,860, L₁=43.243): `σ₀√L₁/ε = 14.44`; `σ₀²L₁/ε² = 208.53`. The
swept ρ grid is {1, 3, 10, 30, 100, 300, 1000} and the smallest swept value
≥ 208.53 is 300. Committed TS-Explore medians across that grid:
1,232,012 / 1,111,759 / 1,158,371 / 1,177,478 / 1,120,860 / 546,610 / 41,058.
Figure caption (paper line 523) reads "The dotted line marks ρvar(ε) from
Eq. (5)".

### A ρ_diag identity

For `V_0 = ρL_G + ρ_diag·I`, the pre-pull effective sample size is
`t_eff,i(0) = 1/[V_0⁻¹]_ii`. Measured on the K=31 SBM, this equals
`K·ρ_diag` exactly (verified at four ρ values under both ρ_diag policies).
Under `ρ_diag = max(1e-4, 1e-6ρ)`:

| ε | ρ = `rho_star` | free t_eff | ρ = `σ₀²L₁/ε²` | free t_eff |
|---|---|---|---|---|
| 1e0 | 43.2 | 3.1e-3 | 1,862 | 5.8e-2 |
| 1e-2 | 4,316 | 1.3e-1 | 1.862e7 | 5.8e2 |
| 1e-3 | 4.316e4 | 1.3e0 | 1.862e9 | 5.8e4 |

With fixed `ρ_diag = 1e-4` the free `t_eff` is `31·1e-4 = 3.1e-3` at every ρ,
and `cond(V_0)` over the ratio grid above ranges 3.7e6 to 6.1e10.

---

## 3. Influence factor: three expressions (Task 5a)

Three expressions are in play:

1. **Thaker et al.** (`Content/Appendix_Resub_Arxiv/influence_factor.tex`),
   verbatim:
   `𝔍(j,D) = min_{i∈C_j(D), i≠j} {r_D(i,j)^{-1}}` if `|C_j(D)|>1`, else 0.
   Equivalently `1/max_i r(i,j)`; independent of `a*`.
2. **Asserted in the rebuttal:** `J(i,G) = ρ/[L_G^†]_ii`.
3. **Computed by `experiments/utils/hardness.py`:** `influence_factors` returns
   the resistance-distance matrix `R`, and `graph_hardness` /
   `competitive_set` consume `J_i = R[i, a*]`.

Coefficient of variation of the ratio (2)/(1) across nodes; 0 means exactly
proportional (any ρ):

| family | CV |
|---|---|
| complete K₂₀ | 0.000 |
| 3-reg clique-union | 0.000 |
| 4-reg random | 0.017 |
| 3-reg random | 0.104 |
| MovieLens K=20 | 0.153 |
| clustered_chain K=20 | 0.216 |
| SBM K=31 (connected) | 0.219 |
| path | 0.228 |
| BA hub-optimal n=20 | 0.547 |
| star K₁,₁₉ | 1.350 |

CV of the ratio (2)/(1/`R[i,a*]`): 0.000 for complete and clique-union, 0.001
for clustered_chain, 0.135–0.530 elsewhere.

**Competitive-set size** at the stopping time on the connected SBM (30
challengers), TS-Explore at ρ=1.096e4, under the three criteria appearing in
the draft:

| criterion | \|H\| |
|---|---|
| `ρ·J(i,G)/2 < 186·C(t)·L₂(t)/Δ²_{i,c}` | 30 |
| `Δ²_{i,c}·J(i,G) ≤ c₀ε²` (Definition 6, c₀=8) | 18 |
| `ρ·J(i,G) ≤ 1/Δ²_{i,c}` (as in `hardness.competitive_set`) | 0 |

All three are logged per snapshot in the traces.

---

## 4. Corollary 13 sandwich (Task 5b)

Tested chain: `(ρ₂(G)−1)·Δ_max⁻² ≤ H_GF ≤ χ̄(G)·Δ_min⁻² ≤ [n/(1+d_min)]·Δ_min⁻²`,
with `ρ₂` the 2-packing number (exact MILP), `χ̄` the clique-cover number
(exact MILP over maximal cliques), `H_GF` the covering-LP optimum.

Over 20 random non-uniform-gap draws (gaps ~ U[0.05, 1.0]) × 10 graph
families = 200 instances:

* `(ρ₂−1)·Δ_max⁻² ≤ H_GF`: held in 200/200.
* `H_GF ≤ χ̄·Δ_min⁻²`: held in 200/200.
* `χ̄ ≤ n/(1+d_min)`: **failed** on 4 of 10 families, in all 20 draws each
  (the inequality is gap-independent):

| family | χ̄ | n/(1+d_min) |
|---|---|---|
| star K₁,₁₉ | 19 | 10.0 |
| 3-reg random | 9 | 5.0 |
| 4-reg random | 8 | 4.0 |
| MovieLens K=20 | 7 | 3.33 |

It held on complete K₂₀ (1 vs 1.0), path (10 vs 10.0), 3-reg clique-union
(5 vs 5.0), clustered_chain (2 vs 10.0), SBM K=31 (10 vs 15.5), and BA
hub-optimal (10 vs 10.0). In the four failing families, max clique size is
below `1+d_min`.

Uniform-gap check (Δ_min = Δ_max = 0.3, the Table 1 regime): the first two
inequalities held on all 10 families.

Structural table (n=20 where applicable) is printed by
`experiments/graph_params.py --which 5a-struct`.

---

## 5. H_GF vs the Russo et al. characteristic time (Task 5c)

`T*` computed from Russo, Song & Pacchiano (AISTATS 2025, arXiv 2503.07824)
Theorem 1, Gaussian specialisation, verbatim from `sections/lower_bound.tex`:

```
T*(nu) = inf_{omega in Delta(V)} max_{u != a*} (m_u^{-1} + m_{a*}^{-1})
             2 lambda^2 / Delta_u^2,   s.t.  m = G^T omega
```

solved with CVXPY/CLARABEL, λ=1.

| family | n | regular | H_GF | T* | H_GF/T* | ratio/log n |
|---|---|---|---|---|---|---|
| empty | 20 | yes | 211.11 | 638.17 | 0.331 | 0.110 |
| complete K₂₀ | 20 | yes | 11.11 | 44.44 | 0.250 | 0.084 |
| star K₁,₁₉ | 20 | no | 11.11 | 44.44 | 0.250 | 0.084 |
| path | 20 | no | 77.78 | 264.42 | 0.294 | 0.098 |
| 3-reg clique-union | 20 | yes | 55.56 | 200.00 | 0.278 | 0.093 |
| 3-reg random | 20 | yes | 54.99 | 181.01 | 0.304 | 0.101 |
| 4-reg random | 20 | yes | 43.93 | 146.53 | 0.300 | 0.100 |
| clustered_chain | 20 | no | 11.11 | 44.44 | 0.250 | 0.084 |
| SBM K=31 | 31 | no | 118.39 | 438.30 | 0.270 | 0.079 |
| BA hub-opt | 20 | no | 44.44 | 128.33 | 0.346 | 0.116 |
| MovieLens K=20 | 20 | no | 214.84 | 861.23 | 0.249 | 0.083 |

Range of `H_GF/T*`: [0.249, 0.346]. `T*` carries a factor `2λ²` and an
`1/m_{a*}` term that `H_GF` does not; when `m_{a*} = m_u` the two expressions
differ by exactly 4.

Consistency check: `H_GF` recomputed in normalised form
(`min_ω max_u 1/(m_u Δ_u²)`) matched `hardness.graph_feedback_hardness` to
displayed precision on all 11 families.

---

## 6. TaS-FG as a live baseline (Task 6)

`experiments/utils/tas_fg.py` implements the informed case with a known
deterministic graph and Gaussian rewards: MLE estimator over side
observations, `ω*(t)` from the plug-in convex program, **averaged D-tracking**
(their Proposition 3), and the GLR stopping rule. For Gaussians their
statistic `L(t) = t·T(N(t)/t; ν̂(t))⁻¹` reduces to a closed form requiring no
convex solve:

```
L(t) = min_{u != a_hat}  Delta_hat_u(t)^2
                         / ( 2 lambda^2 (1/M_u(t) + 1/M_{a_hat}(t)) )
```

Threshold, their Eq. (7): `β(t,δ) = 2·C_exp(ln((K−1)/δ)/2) + 6 ln(1+ln t)`
with `C_exp(x) ≈ x + 4 ln(1+x+√(2x))` for x ≥ 5. At K=20, δ=1e-3 the argument
is x = 4.93, marginally below the stated x ≥ 5 range; the approximation was
used anyway. `β(t=1e4) = 41.43`. The `ω*` solve is cached and re-run every 25
rounds (`resolve_every=1` reproduces their algorithm exactly).

ER n=20, gap 0.3, 5 seeds, δ=1e-3 (`experiments/tas_fg_run.py`). Median
stopping time; **0 cap-hits and 100% correctness in all 15 cells**:

| algorithm | p=0.2 | p=0.4 | p=0.6 |
|---|---|---|---|
| TaS-FG | 10,552 | 4,386 | 3,236 |
| TS-Explore-GF | 24,979 | 9,599 | 7,346 |
| UCB+cover | 16,087 | 8,361 | 6,043 |
| TS+width | 28,153 | 22,945 | 9,669 |
| UCB-N | 27,751 | 16,270 | 9,655 |

`H_GF` = 66.7 / 27.8 / 16.7 at p = 0.2 / 0.4 / 0.6. Same numbers as `T/H_GF`:

| algorithm | p=0.2 | p=0.4 | p=0.6 |
|---|---|---|---|
| TaS-FG | 158.3 | 157.9 | 194.2 |
| TS-Explore-GF | 374.7 | 345.6 | 440.8 |
| UCB+cover | 241.3 | 301.0 | 362.6 |
| TS+width | 422.3 | 826.0 | 580.1 |
| UCB-N | 416.3 | 585.7 | 579.3 |

Wall-clock per round, single-threaded, same runs (ms):

| algorithm | p=0.2 | p=0.4 | p=0.6 |
|---|---|---|---|
| TaS-FG | 1.575 | 1.536 | 1.667 |
| TS-Explore-GF | 0.189 | 0.188 | 0.197 |
| UCB+cover | 0.049 | 0.051 | 0.053 |
| TS+width | 0.158 | 0.162 | 0.157 |
| UCB-N | 0.019 | 0.022 | 0.023 |

TaS-FG's per-round cost here is with `resolve_every=25`; at the paper's
`resolve_every=1` the `omega^*` solve would run 25× as often.

An earlier 3-seed inline pilot on the same instance gave TaS-FG 11,149 /
TS-Explore-GF 38,151 / UCB+cover 27,335 at p=0.2 and 2,824 / 7,544 / 6,043 at
p=0.6, with `T/H_GF` 167.2 and 169.4 for TaS-FG.

**Solver caveat.** CVXPY emits `UserWarning: Solution may be inaccurate` during
some `omega^*` solves. Frequency, measured over 60 solves with perturbed
plug-in gaps representative of mid-run states (ER n=20, p=0.2): **1 of 60**.
`solve_omega_star` falls back to the uniform allocation on an exception, but a
solution flagged inaccurate (which is not an exception) is used as returned.

An attempt to quantify how far the returned `omega^*` is from optimal was
**inconclusive** and its numbers are not reported: the intended reference solve
(SCS at `eps=1e-9`) attained a *worse* objective than CLARABEL in the median
case (ratio 0.95), so it does not serve as ground truth, and the objective
evaluates to `inf` whenever the candidate allocation leaves some `m_u = 0`. The
magnitude of any sub-optimality is therefore unmeasured. The stopping rule is
closed-form and does not involve the solver.

Arithmetic note for calibration: with `T* ≈ 4·H_GF` (from the table in §5) and
`β ≈ 41`, `T* · β / H_GF ≈ 164`.

---

## 7. Smoothness + graph feedback (Task 2)

`experiments/utils/feedback_reg.py` implements the 2×2×2 design in one class:
estimator {`emp`, `reg`} × stop {`ts`, `ucb`} × pull {`cover`, `width`}.

* `emp`: `μ̂_i = R_i^fb/N_i^fb`, `t_eff,i = N_i^fb`.
* `reg`: `V_t = Σ_{s≤t} Σ_{j∈N⁺(π_s)} e_j e_jᵀ + ρL_G + ρ_diag I`,
  `μ̂ = V_t⁻¹ Σ_s Σ_j e_j r_{s,j}`, `t_eff,i = 1/[V_t⁻¹]_ii`.
* UCB radius: noise term `σ√(2L₁(t))` in **both** estimators; `reg` adds
  `√ρ·ε` using the instance's realized ε. TS stopping uses no radius.

Instance `er_smooth`: challengers 1..n−1 form ER(n−1, p) at mean 1−Δ; best arm
0 attached by `n_bridge=1` edge at mean 1.0. So `ε² = n_bridge·Δ²` exactly and
realized ε = 0.300 at every p (recorded per cell; observed range
[0.300, 0.300] in all of them).

Estimator verification (independent of the sweep): Sherman–Morrison inverse vs
direct `np.linalg.inv`, max abs diff 2.3e-13; `μ̂` vs `np.linalg.solve(V,R)`,
7.7e-11; `t_eff = 1/[V⁻¹]_ii` identity exact; `Σ N_fb` equals
`Σ_a pull_counts[a]·|N⁺(a)|`. Mean deviation `|E[μ̂]−μ|` over 40 replications:
0.0338 for `emp` (max at arm 2), 0.1802 for `reg` at ρ=100 (max at arm 0).

n=20, 5 seeds, max_steps 3e6. **Zero cap-hits and 100% correctness in every
cell.** TS/UCB stopping-time ratio at matched pull rule:

**pull = cover**

| p | emp | ρ=1 | ρ=10 | ρ=100 | ρ=1000 |
|---|---|---|---|---|---|
| 0.1 | 1.119 | 1.333 | 0.995 | 0.644 | 0.218 |
| 0.2 | 1.297 | 1.108 | 0.816 | 0.512 | 0.165 |
| 0.4 | 0.884 | 1.229 | 0.745 | 0.462 | 0.102 |
| 0.6 | 1.000 | 0.952 | 0.951 | 0.611 | 0.122 |
| 0.8 | 1.118 | 0.991 | 0.836 | 0.463 | 0.154 |
| 1.0 | 1.722 | 1.478 | 1.372 | 0.706 | 0.386 |

**pull = width**

| p | emp | ρ=1 | ρ=10 | ρ=100 | ρ=1000 |
|---|---|---|---|---|---|
| 0.1 | 1.391 | 1.306 | 0.973 | 0.729 | 0.347 |
| 0.2 | 1.203 | 1.094 | 1.002 | 0.665 | 0.256 |
| 0.4 | 0.958 | 1.125 | 0.831 | 0.559 | 0.161 |
| 0.6 | 1.035 | 1.014 | 0.844 | 0.582 | 0.126 |
| 0.8 | 0.972 | 1.099 | 0.880 | 0.524 | 0.159 |
| 1.0 | 1.125 | 1.028 | 0.868 | 0.542 | 0.312 |

At p=1.0 the graph is complete on the challengers (H_GF = 11.1) and the best
arm has its single bridge edge.

Absolute medians at p=0.2, pull=cover: TS 27,494 / 21,945 / 21,568 / 15,785 and
UCB 24,806 / 26,894 / 42,128 / 95,464 for ρ = 1 / 10 / 100 / 1000. The `emp`
values at p=0.2 are TS 25,746 and UCB 19,848.

Comparison of UCB's ρ-dependence against the radius expression: with
`T(ρ)/T(1) = [(σ√(2L₁)+√ρ·ε)/(σ√(2L₁)+ε)]²`, measured/predicted is

| p | ρ=10 | ρ=100 | ρ=1000 |
|---|---|---|---|
| 0.1 | 1.08 | 1.14 | 1.17 |
| 0.2 | 0.94 | 0.97 | 0.88 |
| 0.4 | 0.88 | 0.81 | 1.28 |

Two design choices, recorded for the reader to weigh: UCB is supplied the
**true** ε in its radius (TS uses no ε); and an earlier version of this
experiment used `σ₀=2σ√14` for `reg` and `σ√2` for `emp`, a 5.3× radius
difference, which was removed so the noise term is identical in both arms.

Figure: `experiments/outputs/fb_smooth.pdf` (regenerate with `experiments/fb_smooth_plot.py`). All six densities complete.

---

## 8. Per-arm instrumentation and pooling factor (Task 1)

`experiments/utils/tracing.py` records, per snapshot: `t_eff,i`, per-arm direct
pulls, `N_i^fb`, `μ̂`, candidate-set membership, competitive-set membership
under all three criteria in §3, plus the disagreement pair and pulled arm per
round, and cap/correctness flags. Opt-in via
`runners.run_algorithm(trace_spec=...)`; verified to leave the RNG stream
bit-identical (stopping times matched exactly for all 7 algorithms tested).

Pooling factor = `t_eff,i / t_i` at termination, over arms with `t_i ≥ 1`.
TS-Explore on the clustered chain at ρ=100, 3 seeds:

| K | pooling over challengers | pooling at a* | \|clique\| | deg(a*) | deg(clique) |
|---|---|---|---|---|---|
| 20 | 2.26 | 1.01 | 19 | 1 | 19 |
| 50 | 11.99 | 1.01 | 49 | 1 | 49 |
| 200 | 141.12 | 1.01 | 199 | 1 | 199 |

Median stopping time and median pooling factor for all three traced algorithms
on the same cells:

| K | TS-Explore | Basic TS | KL-LUCB |
|---|---|---|---|
| 20 | 33,218 (×2.26) | 70,133 (×1.00) | 45,184 (×1.00) |
| 50 | 25,584 (×11.99) | 195,892 (×1.00) | 112,137 (×1.00) |
| 200 | 15,291 (×141.12) | 1,097,211 (×1.00) | 529,706 (×1.00) |

Basic TS and KL-LUCB give pooling = 1.00 exactly, as they use direct counts.
On this instance `|clique| = deg = K−1` for challengers, so cluster size and
degree are not separable here.

Connected SBM (K=31), TS-Explore, 3 seeds, correlation of per-arm pooling
against three candidate predictors:

| ρ | median pooling | max-clique size | degree | influence factor `J` (Thaker def.) |
|---|---|---|---|---|
| 100 | 1.32× | +0.260 / +0.048 | +0.388 / +0.204 | −0.211 / −0.060 |
| 1.096e4 | 19.40× | −0.038 / +0.119 | +0.381 / +0.308 | +0.655 / +0.627 |

(Pearson / Spearman.) Pooling by SBM block (block 0 = the isolated best arm):
at ρ=100, `1.0, 1.0, 1.1, 1.3, 1.4, 1.9`; at ρ=1.096e4,
`1.1, 5.3, 16.0, 29.9, 28.2, 18.7`.

---

## 9. Residual smoothness estimator and doubling schedule (Task 3)

Two estimators implemented in `experiments/utils/adaptive_eps.py`:

* `plugin`: `ε̂² = ⟨μ̂, L_G μ̂⟩` from the operating (regularized) estimator, as
  specified in the plan.
* `probe`: re-solve at a fixed `ρ_probe` decoupled from the operating ρ, and
  subtract the noise contribution:
  `ε̂² = ⟨μ_p, L μ_p⟩ − σ² tr(L V_p⁻¹ N V_p⁻¹)` with
  `V_p = N + ρ_probe L + ρ_diag I`.

Two inflation terms for the certified lower bound `ε_lo = ε̂ − infl`:

* `coordwise`: `√λ_max(L)·‖c(t)‖₂` with
  `c_i = (σ₀√L₁ + √ρ·ε̄)/√t_eff,i`.
* `vnorm`: from `V_t ⪰ ρL_G` applied twice,
  `infl = σ√(2K L₁(t)/ρ) + ε̄`.

Measured `ε̂` from the `plugin` estimator on the K=31 SBM (ε_true = 0.4123),
6,000 pulls, TS-Explore at the stated operating ρ:

| ρ | 1 | 1e2 | 1e4 | 7e5 |
|---|---|---|---|---|
| `ε̂` | 0.9746 | 0.4341 | 0.1114 | 0.0019 |
| `ε̂/ε_true` | 2.364 | 1.053 | 0.270 | 0.005 |

With `plugin`, at ε̄₀ = ε_true/8 (ρ = 7.0e5): `ε̂ = 0.0019`,
`infl = 0.1061` (`vnorm`) or `18.36` (`coordwise`), so `ε_lo = 0` in both
cases and **0 doublings fired** in every configuration tested.

With `probe` (ρ_probe = 1) and `vnorm`, seed 0, `check_every=500`:

| ε̄₀/ε_true | doublings | ε̄_end/ε_true | T | T / T(ε̄=ε_true) |
|---|---|---|---|---|
| 0.125 | 3 | 1.000 | 114,564 | 1.01 |
| 0.25 | 2 | 1.000 | 114,564 | 1.01 |
| 0.5 | 0 | 0.500 | 130,828 | 1.15 |
| 1.0 | 0 | 1.000 | 114,564 | 1.00 |

Doubling trace at ε̄₀/ε_true = 0.125: `t=1500: 0.0515→0.1031 (ε_lo=0.359)`,
`t=1500: 0.1031→0.2062 (ε_lo=0.256)`, `t=2000: 0.2062→0.4123 (ε_lo=0.398)`.
The firing condition `ε_lo > ε̄` with the `vnorm` inflation is equivalent to
`ε̂ − σ√(2KL₁/ρ) > 2ε̄`. The `probe` estimate returned 0 in some
configurations (it is clipped at 0), which suppresses doubling rather than
triggering it. 5-seed sweep in progress.

---

## 10. Feedback sweep at larger n (Task 4a)

ER G(n, 0.2), gap 0.3, 5 seeds, `emp` estimator. `L₂(T) = log(12K²T²/δ)`.
All cells 100% correct, no cap-hits.

| combo | n | T | H_GF | T/H_GF | L₂(T) | T/(H_GF·L₂) |
|---|---|---|---|---|---|---|
| TS+cover | 20 | 24,979 | 66.7 | 374.7 | 35.64 | 10.514 |
| | 50 | 26,382 | 51.1 | 516.5 | 37.58 | 13.744 |
| | 100 | 36,543 | 56.1 | 651.5 | 39.62 | 16.444 |
| UCB+cover | 20 | 16,087 | 66.7 | 241.3 | 34.76 | 6.943 |
| | 50 | 16,227 | 51.1 | 317.7 | 36.61 | 8.678 |
| | 100 | 19,221 | 56.1 | 342.7 | 38.33 | 8.939 |
| TS+width | 20 | 28,153 | 66.7 | 422.3 | 35.87 | 11.771 |
| | 50 | 38,556 | 51.1 | 754.8 | 38.34 | 19.689 |
| | 100 | 83,830 | 56.1 | 1494.4 | 41.28 | 36.206 |
| UCB+width | 20 | 27,751 | 66.7 | 416.3 | 35.85 | 11.613 |
| | 50 | 23,552 | 51.1 | 461.1 | 37.35 | 12.344 |
| | 100 | 58,731 | 56.1 | 1047.0 | 40.56 | 25.810 |

Growth factors n=20→100 in `T/(H_GF·L₂)`: TS+cover ×1.565, UCB+cover ×1.288,
TS+width ×3.076, UCB+width ×2.223. `log n` grows ×1.537 over the same range;
`log n` itself is 3.00 / 3.91 / 4.61.

`H_GF` is non-monotone in n here (66.7 / 51.1 / 56.1) because each n draws a
fresh ER realization. Component counts: n=20 has 2 connected components, n=50
and n=100 have 1. TS+cover and UCB+cover use the same greedy max-cover pull
rule; TS+width and UCB+width use the same argmax-width pull rule.

---

## 11. MovieLens scale extension (Task 4b)

Instance properties, computed without running any bandit (top-k=5,
min_common=5, empirical-rating rewards):

| K | edges | d_min | ε | Δ_min | H_classical | median rating std |
|---|---|---|---|---|---|---|
| 20 | 68 | 5 | 3.408 | 0.0687 | 666.1 | 0.960 |
| 50 | 184 | 5 | 4.523 | 0.0212 | 2,566.9 | 0.971 |
| 100 | 377 | 5 | 6.050 | 0.0097 | 13,657.6 | 0.968 |

K=20 reproduces the paper's reported ε ≈ 3.41 and Δ_min ≈ 0.07.

Committed K=20 medians at ρ=1: Basic TS 1,206,986; KL-LUCB 195,146; GRUB
7,402,007. Scaling those by `H_classical` (666.1 → 2566.9 → 13657.6) gives, as
an extrapolation only: at K=50, Basic TS 4.65e6, KL-LUCB 7.52e5, GRUB 2.85e7;
at K=100, Basic TS 2.47e7, KL-LUCB 4.00e6. The per-run budget is 1e7.

The K=50 run in progress covers TS-Explore, Basic TS, KL-LUCB at
ρ ∈ {1, 100, 1000}, 3 seeds. **GRUB was not included in it**, on the basis of
the extrapolation above; that is a coverage gap in this cell, recorded here
rather than left implicit. K=100 was not attempted.

`movielens_1.py` previously wrote to `movielens_1_results.npz` for any `--K`,
which a K≠20 run would have overwritten. An `--out-tag` argument was added and
K≠20 now auto-tags the filename; K=20 keeps the original name. The committed
file's md5 was checked before and after the K=50 launch and is unchanged.

---

## 12. Flags added so both readings of a mismatch are runnable

Neither default was changed.

* `algobase.AlgoBase(grub_bias=...)`, `grub_log=...` — see §1. Default
  `'legacy'`, verified bit-exact against HEAD.
* `graph_algo.GraphFeedbackTS(pull_scope=...)` — `'all'` searches every arm for
  `argmax_a |N⁺(a) ∩ {î,ĩ}|` (the behaviour described in H.1 line 1091);
  `'pair'` restricts the pull to the disagreement pair (the assumption stated in
  Lemma 11, paper line 869). Default `'all'`. A 2-seed probe at n=20, p=0.2 gave
  `pair` 31,972 and `all` 38,300; not enough seeds to separate.
* `experiments/eps_sensitivity.py --rho-diag-policy {fixed,scaled}` and
  `--residual {probe,plugin}`.

---

## 13. Environment observations

* `RuntimeWarning: divide by zero / overflow / invalid encountered in matmul`
  appears in many logs. `np.ones((32,32)) @ np.ones(32)` under
  `np.seterr(all='raise')` reproduces it on this machine (numpy 2.0.2 built
  against Apple Accelerate, macOS arm64). It appears only for matmuls with
  dimension ≳32, where numpy dispatches to BLAS, and is attributed to whichever
  line next checks the FPU flags. `mu @ L @ mu` on the K=50 chain returns
  0.0899999999999149 against an exact edge-sum reference of
  0.09000000000000002.
* The committed `main_2_results.npz` GRUB row at K=10 is
  `804542, 849561, 834647`. This machine produces `803738, 844702, 832562` from
  identical code and seeds. A pristine `git archive HEAD` checkout produces the
  same values as this machine, i.e. the difference is not from any edit made
  here. Differences are 0.1–0.6%.
* `data/` is gitignored; MovieLens-100K was re-downloaded locally.
* Cross-checkout regression comparison between a pristine `git archive HEAD`
  tree and the current tree, seeds 0–1, same instances (chain K=20 at ρ=100;
  ER n=20 p=0.2):

  | algorithm | HEAD | current | identical |
  |---|---|---|---|
  | TS-Explore | 34581, 33218 | 34581, 33218 | yes |
  | KL-LUCB | 39830, 49508 | 39830, 49508 | yes |
  | TS-GF | 30154, 46445 | 30154, 46445 | yes |
  | TS+width | 36564, 49780 | 36564, 49780 | yes |
  | UCB-N | 32467, 30984 | 32467, 30984 | yes |
  | UCB+cover | 26343, 27335 | 26343, 27335 | yes |

  Together with the `grub_bias='legacy'` check in §1 (also bit-exact), the
  edits made here leave every pre-existing algorithm's output unchanged on
  these cells.

---

## 14. Where the raw data for each section is

Every number above is re-derivable from a saved file without rerunning.
Sections marked "recompute" are pure linear algebra / LP and take seconds.

| § | Raw data | How to regenerate the table |
|---|---|---|
| 1 | `outputs/grub_bias_{chain,movielens}_{legacy,published,sqrt,none}_results.npz` | `python experiments/grub_bias_plot.py` |
| 1 (radius decomposition) | recompute | `python experiments/grub_audit.py --Ks 10 50 200` |
| 2 | `outputs/eps_sensitivity_fixed_results.npz`, `outputs/mis_2_results.npz` (committed), `outputs/movielens_1_results.npz` (committed) | `python experiments/eps_sensitivity.py --seeds 5` (resumes) |
| 3 | recompute | `python experiments/graph_params.py --which 5a` |
| 3 (competitive-set sizes) | `outputs/traces/sbm31_rho10960_TS-Explore_seed*.npz` | `python experiments/trace_runs.py --instance sbm31 --rho 10960` |
| 4 | recompute | `python experiments/graph_params.py --which 5b --gap-seeds 20` |
| 5 | recompute | `python experiments/graph_params.py --which 5c` |
| 6 | `outputs/tas_fg_n20_results.npz` | `python experiments/tas_fg_run.py --ps 0.2 0.4 0.6 --seeds 5` |
| 7 | `outputs/fb_smooth_er_smooth_n20_results.npz` | `python experiments/fb_smooth_plot.py` |
| 8 | `outputs/traces/chain_K{20,50,200}_*_seed*.npz`, `outputs/traces/sbm31_rho{100,10960}_*_seed*.npz` | `python experiments/trace_analysis.py` |
| 9 | `outputs/eps_sensitivity_fixed_adaptive_probe_results.npz` | `python experiments/eps_sensitivity.py --seeds 5 --adaptive --residual {probe,plugin}` |
| 10 | `outputs/fb_scale_p0.2_results.npz` | rerun `experiments/fb_scale.py` (resumes and reprints) |
| 11 | `outputs/movielens_1_K50_results.npz` (in progress); instance table is recompute | `experiments/movielens_1.py --K 50` |
| 13 | n/a | see §13 for the reproduction commands |

§6 now has a persisting runner (`tas_fg_run.py`) and its table above is from
the saved `.npz`. §9's doubling-schedule numbers are still the seed-0 inline
figures: the persistence patch to `eps_sensitivity.py --adaptive` is in place
but that job was stopped before reaching the adaptive stage, so
`eps_sensitivity_fixed_adaptive_probe_results.npz` does not exist yet. It will
be produced by `run_server.sh`.

## 15. Task status

| Task | State |
|---|---|
| 0 GRUB radius | chain K=10–100 done for all four conventions; MovieLens through ρ=300 (`none` through ρ=1000); chain K=200 and the last MovieLens cells running |
| 1 Instrumentation | done |
| 2 Smoothness × feedback | done, all six densities |
| 3 ε sensitivity | 4 of 8 ratio cells done (ratios ≤ 1); ratios 2–16 and the adaptive sweep pending |
| 4a Feedback at larger n | done |
| 4b MovieLens K=50 | running, GRUB excluded (see §11) |
| 5a/5b/5c Graph parameters | done |
| 6 TaS-FG | pilot done at n=20, p ∈ {0.2, 0.6}, 3 seeds |
