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
| Graph-smooth | `thm:main-graph` | `main_*.py` | **CURRENT FOCUS** |
| Asymptotic tuning | `thm:main-mis` + `cor:eps-limit` | `mis_*.py` | Partial — `mis_1.py` produces a U-shape; flagged for resolution |
| Graph feedback | `thm:correct-fb` / `thm:main-fb` | `fb_*.py` | Promising — `fb_1.py` is clean on H_GF tracking; needs UCB-N / UCB-MaxN baselines (Caron et al. 2012) before reviewer-acceptable |
| PSD kernel | `thm:kernel-ts` | `kernel_*.py` | Pending — old exp4 was a no-signal n=20 BA; redo properly |

Naming convention: each script is `<theorem>_<n>.py` where `<theorem>` is one
of `main`/`mis`/`fb`/`kernel` and `<n>` increments per experiment within
that theorem. Outputs share the prefix:
`experiments/outputs/<script>.png`, `<script>_results.npz`,
`<script>_sanity.txt` for the probe (if any).

---

## 1. `thm:main-graph` — graph-smooth sample complexity (Figure 1)

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

### Reframe (2026-04-29): Figure 1 is TS-Explore vs GRUB, not vs Basic TS

The original plan was a K-sweep with bounded $H_{\mathrm{graph}}$ vs
linear-in-$K$ $H_{\mathrm{classical}}$, predicting TS-Explore plateaus
while Basic TS grows. Pre-flight + K=20 sanity falsified the second half
of that prediction:

- Designed a union-of-cliques instance with $H_{\mathrm{graph}} = 11.6$
constant in $K$, $H_{\mathrm{classical}} \in [19, 188]$ over
$K \in \{20,\dots,400\}$ (`experiments/main_preflight.py`).
- At K=20, 5 seeds: TS-Explore $t_{\mathrm{med}} = 17243$, Basic TS
$t_{\mathrm{med}} = 17243$ — **literally identical** (`main_sanity.py`).
GRUB and UCB-no-graph timed out at 200K max_steps.

Why the collapse:

1. TS-Explore's pull rule is $\mathrm{argmin}_{i\in\{\hat i,\tilde i\}}N_i$.
With $\Delta_{\mathrm{decoy}}=1.5$ and $\sigma=1$, decoys almost never
become $\tilde i$, so both algorithms only pull arm 0 and the challenger.
Graph regularization is irrelevant to those decisions.
2. The challenger's graph bonus is $\rho J/2 = 1$; it needs ~8500 direct
pulls; +1 from the graph is noise.
3. Lemma 2's $186\,C\,L/\Delta^2$ threshold needs $J \gtrsim 45000$ for
non-competitive elimination at $\rho=1$ — unreachable on any reasonable
graph. The `competitive_set` predicate $\rho J \le 1/\Delta^2$ uses a
unit constant; the actual stopping rule uses ~$10^3$ times larger. So
$H_{\mathrm{graph}}$ is optimistic book-keeping at $\rho=1$, not what
TS-Explore empirically achieves.

This is general: any instance read at $\rho=1$ will have non-competitive
arms TS-Explore wasn't going to pull anyway, so the graph "saving" is
zero in practice. The empirical payoff of graph regularization shows up
only when $\rho$ is tuned (`thm:main-mis`, §2 below).

### What Figure 1 actually shows

The honest empirical reading of `thm:main-graph`: TS-Explore's
**TS-Explore-style agreement stopping rule** dominates GRUB's UCB-LCB
elimination on the same graph instance at the same $\rho$. The graph
machinery is preserved; what moves the needle is the stopping rule. At
fixed $\rho=1$:

- TS-Explore at K=20 converges in ~17K steps.
- GRUB (`MaxVarianceArmAlgo` with `eliminate_arms` from `algobase.py`)
times out at 200K steps; LCB-radius constant is ~$30$ vs TS's ~$2.6$, so
GRUB needs ~$10^2$ times the effective sample count to eliminate.

This is what the paper's introduction already promises: the
"width-factor inefficiency" of CLUCB-style algorithms in the
combinatorial pure-exploration sense (Wang et al. 2022 cited at
`template_revised.tex:328`), carried over to graph-structured problems.

### Plan

1. **Verify GRUB convergence budget at K=20.** Single seed, no max_steps
cap. If GRUB converges within ~$10^6$ steps, K-sweep is feasible. If
~$10^7$, narrow K range.
2. **Write `experiments/main_1.py`.** K-sweep on
`union_of_cliques_with_challenger`. Algorithms: TS-Explore (graph,
$\rho=1$), Basic TS (no graph), GRUB. Optionally NoGraphAlgo (UCB).
$\delta=10^{-3}$, $q=0.1$.
3. **Seeds.** 20 minimum for visible IQR. K range determined by step 1.
4. **Two-panel figure.** (A) Median stopping time vs K, log y-axis,
shaded IQR. (B) Elimination/agreement curves on a single seed at one K
(e.g., K=100), to convey *how* the algorithms differ — TS-Explore
collapses to convergence, GRUB drips arms out.
5. **Acceptance.** TS-Explore $\ge 10\times$ faster than GRUB at every K
in the sweep, with confidence intervals separated. Basic TS may or may
not be on the figure — its presence honestly admits "graph isn't the
win at $\rho=1$"; its absence keeps Figure 1 focused on the GRUB
replacement claim. Decide once we see the curves.

### Done criterion

- A figure with median + IQR for TS-Explore vs GRUB on
`union_of_cliques_with_challenger`, K-sweep, 20+ seeds, $\rho=1$.
- A second panel with elimination/agreement-history at a single K.
- TS-Explore at least $10\times$ faster than GRUB on every K, with no
overlapping IQR.
- A short paper-side note: "graph regularization at $\rho=1$ does not
materially change TS-Explore's empirical stopping time on this family
of instances; the graph payoff appears under $\rho$-tuning, see §2".

Once this lands, move to `thm:main-mis`. The original "K-sweep with
bounded $H_{\mathrm{graph}}$" plan and the algorithmic-modification
fallback are both dropped.

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

## 3. `thm:main-fb` — promising, baselines missing

`experiments/fb_1.py` runs the density sweep on Erdős–Rényi `n=20`, gap
`Δ=0.3`, 7 values of `p ∈ [0.05, 1.0]`, 10 seeds. Outputs
`experiments/outputs/fb_1.png` and `fb_1_results.npz`. Headline
results: `H_GF` shrinks 11× across the sweep and `TS-Explore-GF`'s
median stopping time shrinks by the same factor; at `p=1` (clique)
`TS-Explore-GF` is `16.4×` faster than graph-smooth `TS-Explore` and
`16.6×` faster than `Basic TS`. Endpoints match the closed-form values
in `cor:HGF-bounds`.

**Outstanding before this is reviewer-acceptable:** the only baselines
right now are `TS-Explore` (graph-smooth) and `Basic TS`, both of which
are strawmen for the graph-feedback setting. A reviewer will demand at
least one proper graph-feedback baseline — UCB-N or UCB-MaxN
(Caron et al. 2012), or one of the side-observation algorithms from
Mannor–Shamir / Cohen et al. Implementing one of these is the next step
for `fb_1.py` (or a follow-on `fb_2.py`).

Hold this line of work until `thm:main-graph` is closed.

## 4. `thm:kernel-ts` — not started

Old `exp4_kernel_comparison.py` (in `old/`) was a non-signal on n=20 BA.
Needs a redesign on n≥100 BA before any meaningful comparison between
combinatorial and normalised Laplacian. Waits.
