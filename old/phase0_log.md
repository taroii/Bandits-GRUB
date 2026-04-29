# Phase 0 Status Log

Branch: `paper-experiments`. Working tree is dirty (all new files + `algobase.py` / `graph_algo.py` rewrites, no commits yet).

## Done

- `experiments/` scaffolding (`__init__.py`, `_paths.py`, `outputs/`)
- `experiments/utils/instances.py` — SBM / ER / BA / complete / empty / phase-transition builders
- `experiments/utils/hardness.py` — classical / graph / GF hardness, `rho_star`, `epsilon_hardness`, `critical_epsilons`, competitive-set calculators
- `experiments/utils/runners.py` — seeded + multiprocessing harness
- `experiments/utils/plotting.py` — CI plotting + algo styles
- `algobase.py` refactored: `eta → rho_lap`, `rho → rho_diag`, `epsilon_nominal` override, `kernel` param, Sherman–Morrison-based `update_conf_width`
- `graph_algo.py`: `get_all_teff` uses `diag(inverse_tracker)` (Phase 0.6 fix), new `GraphFeedbackTS` with dominating-set init + max-cover pulling rule
- `base/sample_main.py` + `base/sample_main2.py` updated to new kwargs
- `experiments/phase0_verify.py` — all 5 Phase 0.9 checks
- `experiments/exp1_…exp6_*.py` + `experiments/run_all.py`

## Open issue — needs human check before Phase 1

Phase 0.9 check 1 (hardness) passed. Check 2 (ThompsonSampling regression) is where I got stuck:

- On a small SBM (K=16, δ=10⁻³), the **refactored** `ThompsonSampling` does not converge within 5 000 rounds — it keeps pulling every arm roughly uniformly instead of focusing on `i_hat` vs `i_tilde`. `BasicThompsonSampling` on the same problem converges in ~500.
- I started an apples-to-apples test against the pre-refactor `ThompsonSampling` (via `git show e6ef13b:…`) but hadn't confirmed yet whether the old code also fails — that's the missing data point.

Two possibilities:

- **(a)** I introduced a bug in the refactor (most likely: something about `mean_estimate` shape after removing the `np.matrix` inputs, or the confidence-width update ordering).
- **(b)** The original TS also had this convergence issue on small δ and the old sample runs only worked because they used larger gaps / easier instances.

## Resolution (2026-04-28)

Closed: **(b)** is correct. The refactor is a faithful rename — Phase 0.9 `phase0_verify.py` regression test (K=3 clique, mu=[2.0, 0.2, 0.1], q=0.01) shows OLD and NEW produce *bitwise-identical* stopping times across three seeds: `[88, 112, 60]`. Sherman–Morrison-tracked inverse vs direct `np.linalg.inv` agree to relative error 1e-11 to 1e-13 across rho regimes.

The K=16 / q=0.01 / 5 000-round non-convergence was just a budget mismatch, not a bug:
- At q=0.01, K=16, δ=10⁻³ the algorithm needs `M(δ,q,t) = (1/q)·log(12K²t²/δ) ≈ 100·30 = 3 000` agreeing TS samples per round; with cluster-1 gap 0.387 and t_eff ~ pulls/K, all 3 000 agreeing has near-zero probability until t reaches O(10⁵). At q=0.1 the same instance converges in 36 042 rounds.
- 5 000 rounds was simply far below the natural stopping time. The diagnosis test was using a budget two orders of magnitude too small.

`phase0_verify.py` — 11/12 checks pass. The one fail is the Sherman–Morrison speedup target (≥4× at n=400, observed 1.1×). Cause: the per-round cost is dominated by the `M(δ,q,t)·K`-sized RNG draw, not the inverse, so SM doesn't show its asymptotic advantage at n=400 — the matrix work *is* faster but invisible behind the Gaussian sampling. Not a correctness issue.

## Exp3 results (2026-04-28)

Ran `exp3_smoothness_asymptotics.py --seeds 5 --n-jobs 4 --max-steps 2000000` on the K=31 phase-transition instance. Run time ~70 minutes wall-clock (4 cores).

- **rho-tuning matters (Theorem 4.4 / Cor 4.5):** TS_rho1 t_med = 1 372 441; TS_tuned t_med at eps=10⁻³ = 268 100 → **5.12× speedup** from the optimal rho schedule.
- **Phase transitions visible** between eps=10⁻² and eps=10⁻³: TS_tuned drops from 1.30M → 0.56M → 0.27M as eps drops from 10⁻² → 10⁻²·⁵ → 10⁻³, matching the predicted eps_i★ band (∈ [4.7×10⁻³, 1.4×10⁻³] for the largest-gap clusters).
- **Asymptotic limit not reached:** at the smallest eps in the sweep, TS_tuned t_med = 268 100 vs theoretical max 1/Δ²·log(1/δ) = 691, ratio 388×. The bound has hidden polylog × constants (the 186·C(t)·L(t) prefactor in Lemma 4.3); the *direction* of decrease matches the corollary.

Two of three acceptance criteria pass: monotone-decreasing in eps (✓), rho-tuning ≥2× (5.12×, ✓), within 3× of asymptote (388×, ✗ as expected).

Two open caveats for the writeup:
1. H_ε panel (Panel B) drops by only ~6% across the sweep (710 → 671) because the smallest eps_i★ in this instance is 5.8×10⁻⁵, well below the eps=10⁻³ floor. To see H_ε approach 100 (the max), the sweep needs to extend down to eps ≈ 10⁻⁴.
2. eps_nominal is not used by the TS sampling rule (only by `eliminate_arms`, which TS doesn't call). The TS_tuned vs TS_rho1 distinction therefore reduces to a rho_lap sweep — fine in spirit (Cor 4.5 is about the algorithm's choice of rho), but worth flagging when describing the experiment.

## Extended exp3 sweep (2026-04-28)

Added eps points 10⁻³·⁵, 10⁻⁴, 10⁻⁴·⁵ to expose the full H_ε staircase. Result: H_ε reaches its asymptotic max=100 at eps=3.16×10⁻⁵ (|comp set|=0 → all suboptimal arms non-competitive), confirming Cor 4.5 (i)–(ii) as a *structural* claim.

But the **empirical stopping time is U-shaped in ε**, not monotone:

| log₁₀(ε) |  H_ε  | \|comp\| | TS_tuned t_med |
|---|---|---|---|
| 0.0   | 710 | 30 | 1 299 422 |
| -1.0  | 710 | 30 | 1 299 422 |
| -2.0  | 710 | 30 | 1 185 749 |
| -2.5  | 710 | 30 |   557 498 |
| **-2.5 → -3.0** | 704 → 671 | 24 → 12 | **265 361 → 268 100  (minimum)** |
| -3.5  | 611 |  6 |   362 375 |
| -4.0  | 611 |  6 |   634 435 |
| -4.5  | 100 |  0 | 1 225 829 |

The minimum empirical T (~265k) is achieved at ε ≈ 10⁻²·⁵ to 10⁻³, where H_ε is still 700+ and 12–24 arms are still competitive. Going to smaller ε (where H_ε approaches 100) makes T climb back to ~1.2M.

Hypothesis: at ρ ≫ pulls/K, the Laplacian regularization smooths µ̂ aggressively toward the cluster mean. For the smallest-gap cluster (gap 0.1, J≈1) the bias is on the order of (ρ·ε / √t_eff,i) — which is *bounded* by σ₀√L₁ by construction of ρ⋆, so it shouldn't blow up. But the TS *sampling* rule uses N(µ̂, C/t_eff) with no bias correction, so any small bias in µ̂ that survives the asymptotic balance dominates the tiny TS variance, and the M-sample agreement rule fails.

**Implication for the paper.** Theorem 4.4's bound `T = O(H_ε · polylog(K, 1/δ))` is correct but increasingly *loose* as ε → 0. The empirical optimum ε is *not* the asymptotic ε → 0; instead it sits at a "sweet spot" where the graph already collapses many small-gap arms (large reduction from sum to staircase) without driving the regularizer so hard that µ̂ bias swamps the TS sampling variance.

Worth a one-paragraph remark in §4.2 of the paper: the *bound* is monotonically decreasing in ε, but the *algorithm* has a finite-sample sweet spot. The bound's polylog and constants get worse as ρ⋆ grows — Lemma 4.3 has a 186·C(t)·L(t) prefactor that becomes the dominant term once H_ε is small. Reporting both the bound's prediction and the empirical curve is a more honest pitch than just "ε → 0 is best."

## Exp1 sanity check (2026-04-28)

Probed several instances to find one where Theorem 4.1's `H_graph << H_classical` separation is empirically visible at the default ρ=1. **No instance shows it.**

| instance | K | H_classical | H_graph(ρ=1) | ratio | TS t_med | Basic t_med | empirical ratio |
|---|---|---|---|---|---|---|---|
| Path n=30, mixed gaps | 30 | 101.9 | 12.2 | 8.4× | 106k | 106k | 1.00× |
| Path n=30, uniform 0.3 gap | 30 | 322.2 | 133.3 | 2.4× | 109k | 109k | 1.00× |
| K=11 SBM (current exp1) | 11 |  59.0 |  56.2 | 1.05× | ~100k | ~100k | 1.00× |
| K=101 SBM (notes target) | 101 | 315.0 | 294.8 | 1.07× | (large) | (large) | (untested) |

The TS-Explore algorithm at ρ=1 is empirically **indistinguishable from Basic TS**, even when the bound predicts an 8× speedup. The reason is structural: TS-Explore stops when all M(δ,q,t) Thompson samples agree on i_hat. This requires the variance C/t_eff_i for the smallest-gap suboptimal arm to be small enough that no sample picks it. The graph contribution to t_eff (lower bound `t_eff,i ≥ t_i + ½·min(ρ·J(i,G), Σ pulls in C(i))`) is at most O(ρ·J), which is O(1) at ρ=1. The dominant term in the closest-gap arm's t_eff is its direct pull count t_i, which scales as 1/Δ_min² regardless of graph.

**Implication:** Theorem 4.1's H_graph hardness is a valid *upper bound*, but the TS-Explore algorithm's actual stopping time is bottlenecked by the smallest gap (classical 1/Δ_min² scaling), not by H_graph. The benefit predicted by H_graph (collapsing the sum over competitive arms to a single max for non-competitive ones) doesn't materialize for the closest competitor, which is always competitive.

This is consistent with the U-shape we observed in exp3: the only way to push t_eff_i for the *closest competitor* to be ≫ 1/Δ_min² is via huge ρ — but that hits the bias wall first.

**For exp1 to validate Theorem 4.1**, we would need either:
- An algorithm whose stopping rule directly uses H_graph (e.g., the elimination-rule UCB-style algorithms `MaxDiffVarAlgo` etc., assuming they also use the graph-aware confidence bounds), not the TS-Explore agreement rule.
- Acceptance of a weaker claim: "TS-Explore stopping time is no worse than Basic TS, and dramatically faster than Basic TS in the large-ρ regime (Theorem 4.4)."

The cleanest empirical story is therefore exp3 (Thm 4.4 / Cor 4.5) and exp2 (Thm 4.8), not exp1 (Thm 4.1). The paper would benefit from re-orienting §5 (experiments) around the rho-tuning narrative rather than rho=1 H_graph claims.

## What to run next

Three commands, in order. Each should finish in under a minute.

```bash
cd /Users/taro/Documents/Bandits-GRUB

# 1. Pre-refactor TS on the same small instance — does the OLD code converge?
mkdir -p /tmp/old_grub && \
  git show e6ef13b:graph_algo.py  > /tmp/old_grub/graph_algo.py && \
  git show e6ef13b:algobase.py    > /tmp/old_grub/algobase.py  && \
  git show e6ef13b:support_func.py> /tmp/old_grub/support_func.py && \
  cp graph_generator.py /tmp/old_grub/ && \
python -u -c "
import numpy as np, sys
sys.path.insert(0, '/tmp/old_grub')
import graph_algo, graph_generator as gg
np.random.seed(0)
G = gg.call_generator(5, 3, 0.9, [0.9, 0.5, 0.3], 'SBM', q=0.0)
A = np.asarray(G['Adj']); D = np.asarray(G['Degree']); mu = np.asarray(G['node_means']).copy()
mu[0] = 1.3 * mu.max()
np.random.seed(42)
ts = graph_algo.ThompsonSampling(D, A, mu, eta=1.0, delta=1e-3, q=0.01, eps=0.0)
for it in range(100000):
    ts.play_round(1)
    if ts.converged: break
print('OLD: t=', ts.t, 'conv=', ts.converged, 'remain=', ts.remaining_nodes)
"

# 2. Refactored TS on the same instance (bigger budget)
python -u -c "
import numpy as np, graph_algo, graph_generator as gg
np.random.seed(0)
G = gg.call_generator(5, 3, 0.9, [0.9, 0.5, 0.3], 'SBM', q=0.0)
A = np.asarray(G['Adj']); D = np.asarray(G['Degree']); mu = np.asarray(G['node_means']).copy()
mu[0] = 1.3 * mu.max()
np.random.seed(42)
ts = graph_algo.ThompsonSampling(D, A, mu, rho_lap=1.0, delta=1e-3, q=0.01)
for it in range(100000):
    ts.play_round()
    if ts.converged: break
print('NEW: t=', ts.t, 'conv=', ts.converged, 'remain=', ts.remaining_nodes)
"

# 3. Full Phase-0.9 verification (skip if 1 & 2 disagree — regression test will hang otherwise)
PYTHONUNBUFFERED=1 python -u experiments/phase0_verify.py
```

### Interpreting the outcomes

- **OLD also fails to converge** → the issue predates my refactor. The TS stopping rule on this instance just needs a much larger budget; bump `max_steps` in the regression test and move on.
- **OLD converges but NEW does not** → I have a bug. Save the two output lines (or "OLD conv True, NEW conv False") and I'll fix it next session.
- **Both converge** with similar times → Phase 0.9 is good. Try `python experiments/exp1_delta_scaling.py --quick --seeds 3` to smoke-test one experiment, or `python experiments/run_all.py --quick` for the whole suite with tiny seeds.

## Caveats

- Experiments 1–6 and `run_all.py` are scaffolded but **not yet smoke-tested end-to-end** — that depends on TS convergence behaving.
- No commits yet. If you want a checkpoint before debugging, `git add -A && git commit -m "Phase 0 infrastructure WIP"` is safe; all touched files are tracked or under `experiments/`.
