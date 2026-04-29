# Wednesday plan — 2026-04-29

## Direction

**Theorem-by-theorem. One at a time. Don't move on until the current one is
genuinely solved.** "Solved" means a reviewer reading the paper would
acknowledge the experiment validates the theorem on its own terms — not "the
acceptance script's `[x]` boxes pass."

Old infrastructure (six experiments, phase-0 scaffolding, sample scripts,
the bulk of `notes.md` and `phase0_log.md`) is in `old/` for reference; do
not load anything from there into the new line of work without an explicit
reason. The clean repo is now: `algobase.py`, `graph_algo.py`,
`graph_generator.py`, `support_func.py`, `template_revised.tex`, plus the
single live experiment described below.

## Theorem queue (priority order)

| Theorem | Label | Script prefix | Status |
|---|---|---|---|
| Graph-smooth | `thm:main-graph` | `main_*.py` | **CURRENT FOCUS — design from scratch** |
| Asymptotic tuning | `thm:main-mis` + `cor:eps-limit` | `mis_*.py` | Partial — `mis_1.py` produces a U-shape; flagged for resolution |
| Graph feedback | `thm:correct-fb` / `thm:main-fb` | `fb_*.py` | Pending — old exp2 was clean, will revive |
| PSD kernel | `thm:kernel-ts` | `kernel_*.py` | Pending — old exp4 was a no-signal n=20 BA; redo properly |

Naming convention: each script is `<theorem>_<n>.py` where `<theorem>` is one
of `main`/`mis`/`fb`/`kernel` and `<n>` increments per experiment within
that theorem. Outputs share the prefix:
`experiments/outputs/<script>.png`, `<script>_results.npz`,
`<script>_sanity.txt` for the probe (if any).

---

## 1. `thm:main-graph` — graph-smooth sample complexity

### What the theorem says

For $q\in[\delta,0.1]$, with probability $\ge 1-\delta$ TS-Explore returns
$a^\star$ in
\[
T = O\!\left(H_{\mathrm{graph}}\log\tfrac1\delta + H_{\mathrm{graph}}\log^2(KH_{\mathrm{graph}})\right)
\quad\text{(at }q=\delta),
\]
where
$H_{\mathrm{graph}} \asymp \sum_{i\in\mathcal H\setminus\{a^\star\}}\Delta_{i,c}^{-2}
+ \max_{i\in\mathcal N}\Delta_{i,c}^{-2}$.

### What a reviewer wants

A figure where TS-Explore's stopping time scales like $H_{\mathrm{graph}}$
on instances where $H_{\mathrm{graph}} \ll \sum_i\Delta_i^{-2}$, and a head
to head against the algorithm this paper claims to replace
(`GRUB`, Thaker et al. 2022).

### What we already know is hard

At $\rho=1$ on small SBMs and on path graphs, **TS-Explore is empirically
indistinguishable from Basic TS** even when the analytical ratio
$H_{\mathrm{graph}}/H_{\mathrm{classical}}$ is 8×. The TS-Explore stopping
rule (all $M$ samples agree) is bottlenecked by the smallest-gap arm's
sampling variance $C/t_{\mathrm{eff},i}$, and the graph contributes only
$O(\rho\,\mathcal J)$ to that arm's $t_{\mathrm{eff}}$ at $\rho=1$. The
$H_{\mathrm{graph}}$ savings live in the cheaply-eliminated non-competitive
arms, which weren't going to dominate $T$ anyway.

This is the central problem to solve. Two interpretations:

1. **Algorithmic** — the TS-Explore agreement rule cannot expose
$H_{\mathrm{graph}}$ on its own. We need either (a) a different stopping
rule (e.g., Lemma 2's $t_{\mathrm{eff},i} \ge 186\,C(t)L(t)/\Delta_{i,c}^2$
elimination check applied directly), or (b) a different pull rule that
actively elects to skip non-competitive arms.

2. **Instance-design** — the right experiment chooses an instance where the
non-competitive arms *do* dominate $H_{\mathrm{classical}}$, e.g., by
scaling $K\to\infty$ with bounded $H_{\mathrm{graph}}$. The bound
$T = O(H_{\mathrm{graph}}\,\mathrm{polylog})$ then implies $T$ saturates
while Basic TS grows with $K$.

Both of these are testable. Plan below addresses option 2 first because it
keeps the algorithm unchanged; option 1 is a fallback if the K-scaling
experiment doesn't separate the curves.

### Plan

1. **Pick a graph family with bounded $H_{\mathrm{graph}}$ as $K$ grows.**
A union of cliques with one isolated best arm is the cleanest: every
suboptimal arm is non-competitive once $\rho\,\mathcal J(i,G)$ exceeds
$\Delta^{-2}$, so $H_{\mathrm{graph}} = O(\Delta^{-2})$ regardless of $K$.

2. **K-scaling experiment.** Vary $K \in \{20, 50, 100, 200, 400\}$, fix
$\delta=10^{-3}$, $\Delta = 0.3$ for all suboptimals. Plot stopping time vs
$K$ for TS-Explore (graph), Basic TS, and (if reachable) GRUB. The
prediction: TS-Explore plateaus, Basic grows linearly in $K$.

3. **Acceptance.** A reviewer will accept this iff
- the TS curve is sub-linear in $K$ over at least a 5× range, and
- TS is at least $3\times$ faster than Basic TS at the largest $K$.

4. **If step 3 fails on the cleanest instance, we go to option 1**: replace
or augment the TS-Explore stopping/pull rule so that non-competitive arms
are elected for elimination when their lower confidence bound certifies
non-competitiveness. The `eliminate_arms` machinery in `algobase.py`
already does this for the UCB baselines; the question is whether
incorporating it into TS-Explore preserves the proof.

### Done criterion

- A figure (or figures) clearly showing TS-Explore stopping time
governed by $H_{\mathrm{graph}}$ rather than $\sum_i\Delta_i^{-2}$,
- on a single named instance,
- with at least 20 seeds and visible IQR,
- with a side-by-side curve for Basic TS that is materially larger,
- and ideally a curve for GRUB that is comparable or worse.

Until that exists, do not start work on `thm:main-mis`, `thm:main-fb`, or
`thm:kernel-ts`.

---

## 2. `thm:main-mis` + `cor:eps-limit` — flagged for later

`experiments/mis_1.py` runs and produces a clean figure
(`experiments/outputs/mis_1.png`, `mis_1_results.npz`). 10 ε points, 5 seeds,
K=31 SBM. **The structural predictions of the corollary hold exactly**:
$|\mathcal H_\varepsilon| \to 0$ and $H_\varepsilon \to \max_i \Delta_{i,c}^{-2}=100$
in the staircase predicted by Definition 4.3.

**The empirical stopping time is U-shaped**, not monotone. Optimum at
$\varepsilon \approx 10^{-2.5}$ (5.12× speedup over $\rho=1$ baseline);
beyond that, $T$ climbs back toward baseline. This is consistent with the
theorem read as an upper bound — the $186\,C(t)L(t)$ prefactor in
Lemma 2 grows with $\rho^\star$ and eventually outpaces the $H_\varepsilon$
reduction. It is not consistent with `cor:eps-limit`(iii) read as a literal
asymptotic.

### To resolve before publication

- Decide the framing: does `cor:eps-limit`(iii) become an upper-bound
statement only, with the sweet-spot ε reported as the practical
recommendation? Or do we modify the algorithm (e.g., bias-corrected TS
sampling — inflate the variance to absorb the bias term) to recover a
monotonic curve? The former is a paper-edit; the latter is research and
risks breaking the proof.
- Re-render with 20+ seeds once the framing is decided.

Hold this line of work until `thm:main-graph` is closed.

---

## 3. `thm:main-fb` and `thm:kernel-ts` — not started

Old work in `old/experiments/exp2_density_sweep.py` (graph feedback) was
the strongest of the six, but is not loaded. Old `exp4_kernel_comparison.py`
was a non-signal — needs a redesign on n≥100 BA. Both wait.
