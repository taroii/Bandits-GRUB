"""mis_1 — smoothness asymptotics and phase transitions.

Targets ``thm:main-mis`` and ``cor:eps-limit``. Sweeps nominal smoothness
epsilon downward while tuning rho = rho^*(epsilon); expects stopping time
to approach max_i 1/Delta_{i,c}^2 log(1/delta), with discrete drops at
each epsilon_i^*.

The script runs a ``probe_rho_star()`` sanity check before the main sweep
and writes the probe output to ``experiments/outputs/mis_1_sanity.txt``.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.utils import instances, hardness, runners, plotting  # noqa: E402
import graph_algo  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs')
os.makedirs(OUT, exist_ok=True)


def build_L(D, A, kernel='combinatorial'):
    return graph_algo.algobase.build_kernel(D, A, kernel)


# Picklable factories (multiprocessing requires top-level callables).
class TSTunedFactory:
    def __init__(self, D, A, mu, rho_lap, delta, q, epsilon_nominal, rho_diag):
        self.kw = dict(D=D, A=A, mu=mu, rho_lap=rho_lap, delta=delta, q=q,
                       epsilon_nominal=epsilon_nominal, rho_diag=rho_diag)

    def __call__(self):
        return graph_algo.ThompsonSampling(**self.kw)


class TSRho1Factory:
    def __init__(self, D, A, mu, delta, q):
        self.kw = dict(D=D, A=A, mu=mu, rho_lap=1.0, delta=delta, q=q)

    def __call__(self):
        return graph_algo.ThompsonSampling(**self.kw)


class BasicFactory:
    def __init__(self, mu, delta, q):
        self.kw = dict(mu=mu, delta=delta, q=q)

    def __call__(self):
        return graph_algo.BasicThompsonSampling(**self.kw)


def probe_rho_star(mu, A, D, epsilons, delta, sigma=1.0,
                   rho_diag_policy='scaled'):
    """Return a dict of diagnostics for each epsilon in ``epsilons``."""
    K = len(mu)
    T_est = hardness.classical_hardness(mu) * np.log(1 / delta)
    rows = []
    L = build_L(D, A, 'combinatorial')
    for eps in epsilons:
        rho = hardness.rho_star(eps, K, T_est, delta, sigma)
        if rho_diag_policy == 'scaled':
            rho_diag = max(1e-4, 1e-6 * rho)
        else:
            rho_diag = 1e-4
        V0 = rho * L + rho_diag * np.eye(K)
        try:
            cond = float(np.linalg.cond(V0))
        except np.linalg.LinAlgError:
            cond = np.inf
        V0_inv = np.linalg.inv(V0)
        diag = np.diag(V0_inv)
        teff = 1.0 / np.maximum(diag, 1e-300)
        rows.append({
            'eps': eps,
            'rho_star': rho,
            'rho_eps': rho * eps,
            'sigma0_sqrtL1': 2.0 * sigma * np.sqrt(14.0) * np.sqrt(
                np.log(12 * K ** 2 * max(T_est, 1) ** 2 / delta)),
            'cond': cond,
            'rho_diag': rho_diag,
            'teff_min': float(np.min(teff)),
            'teff_median': float(np.median(teff)),
            'teff_max': float(np.max(teff)),
        })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', type=int, default=10)
    parser.add_argument('--n-jobs', type=int, default=1)
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--c0', type=float, default=4.0)
    parser.add_argument('--q', type=float, default=0.1)
    # eps=1.0 needs ~1.3M rounds on the K=31 SBM at q=0.1; budget conservatively.
    parser.add_argument('--max-steps', type=int, default=2_000_000)
    args = parser.parse_args()

    if args.quick:
        eps_sweep = [1.0, 10 ** -1.0, 10 ** -2.0]
        seeds = list(range(max(args.seeds, 3)))
    else:
        eps_sweep = [10 ** (-k / 2.0) for k in range(0, 7)]
        seeds = list(range(args.seeds))
    delta = 1e-3

    mu, A, D = instances.sbm_phase_transition(seed=0)
    K = len(mu)
    print(f"[mis_1] K={K}, gaps^2 = "
          f"{np.sort((mu.max() - mu)[mu != mu.max()] ** 2)[:5]}", flush=True)

    # --- Sanity check --------------------------------------------------------
    probe = probe_rho_star(mu, A, D, eps_sweep, delta)
    probe_lines = ["# mis_1 probe_rho_star output\n"]
    ok = True
    for row in probe:
        line = (f"eps={row['eps']:.3e}  rho*={row['rho_star']:.3e}  "
                f"rho_diag={row['rho_diag']:.1e}  "
                f"cond(V0)={row['cond']:.2e}  "
                f"teff[min/med/max]={row['teff_min']:.2e}/"
                f"{row['teff_median']:.2e}/{row['teff_max']:.2e}  "
                f"rho*eps/[sigma0 sqrtL1] = "
                f"{row['rho_eps']/row['sigma0_sqrtL1']:.4f}")
        probe_lines.append(line)
        if row['cond'] > 1e10:
            ok = False
        if abs(row['rho_eps'] / row['sigma0_sqrtL1'] - 1.0) > 0.01:
            ok = False
    with open(os.path.join(OUT, 'mis_1_sanity.txt'), 'w') as f:
        f.write("\n".join(probe_lines) + "\n")
    print("\n".join(probe_lines), flush=True)
    if not ok:
        print("[mis_1] WARNING: sanity check raised flags - "
              "continuing but results may be unreliable.", flush=True)

    # --- eps_i^* predictions -------------------------------------------------
    T_est = hardness.classical_hardness(mu) * np.log(1 / delta)
    eps_critical = hardness.critical_epsilons(mu, A, D, T_est, delta,
                                              c0=args.c0)
    print("[mis_1] epsilon_i^* values (finite):")
    for i, v in sorted(eps_critical.items(), key=lambda kv: kv[1]):
        if np.isfinite(v):
            print(f"  arm {i:3d}: eps_i^* = {v:.3e}", flush=True)

    # --- Sweep ---------------------------------------------------------------
    # TS_rho1 and Basic are independent of eps (epsilon_nominal is not used by
    # the TS sampling rule, it only enters the elimination bias term in
    # eliminate_arms which TS doesn't call). Compute once per seed and broadcast.
    algo_names = ['TS_tuned', 'TS_rho1', 'Basic']
    stop_times = {name: np.zeros((len(eps_sweep), len(seeds))) for name in algo_names}
    correct = {name: np.zeros((len(eps_sweep), len(seeds)), dtype=bool)
               for name in algo_names}
    H_eps_vals = []
    comp_size = []

    print("[mis_1] running eps-independent baselines once...", flush=True)
    fac_rho1 = TSRho1Factory(D, A, mu, delta=delta, q=args.q)
    fac_basic = BasicFactory(mu, delta=delta, q=args.q)
    t0 = time.time()
    runs_rho1 = runners.run_many(fac_rho1, seeds, n_jobs=args.n_jobs,
                                 max_steps=args.max_steps)
    print(f"  TS_rho1 baseline: t_med={np.median([r['stopping_time'] for r in runs_rho1]):.0f} "
          f"({time.time()-t0:.1f}s)", flush=True)
    t0 = time.time()
    runs_basic = runners.run_many(fac_basic, seeds, n_jobs=args.n_jobs,
                                  max_steps=args.max_steps)
    print(f"  Basic   baseline: t_med={np.median([r['stopping_time'] for r in runs_basic]):.0f} "
          f"({time.time()-t0:.1f}s)", flush=True)
    for si, r in enumerate(runs_rho1):
        stop_times['TS_rho1'][:, si] = r['stopping_time']
        correct['TS_rho1'][:, si] = r['correct']
    for si, r in enumerate(runs_basic):
        stop_times['Basic'][:, si] = r['stopping_time']
        correct['Basic'][:, si] = r['correct']

    for ei, eps in enumerate(eps_sweep):
        rho_star = hardness.rho_star(eps, K, T_est, delta)
        rho_diag_val = max(1e-4, 1e-6 * rho_star)
        H_eps_vals.append(hardness.epsilon_hardness(
            mu, A, D, eps, T_est, delta, c0=args.c0))
        H_set, N_set = hardness.competitive_set_epsilon(
            mu, A, D, eps, T_est, delta, c0=args.c0)
        comp_size.append(len(H_set))
        print(f"[mis_1] eps={eps:.3e} rho*={rho_star:.3e} H_eps={H_eps_vals[-1]:.2f} "
              f"|comp|={len(H_set)}", flush=True)

        fac_tuned = TSTunedFactory(D, A, mu, rho_lap=rho_star, delta=delta,
                                   q=args.q, epsilon_nominal=eps,
                                   rho_diag=rho_diag_val)
        t0 = time.time()
        runs = runners.run_many(fac_tuned, seeds, n_jobs=args.n_jobs,
                                max_steps=args.max_steps)
        print(f"    TS_tuned: t_med={np.median([r['stopping_time'] for r in runs]):.0f} "
              f"({time.time()-t0:.1f}s)", flush=True)
        for si, r in enumerate(runs):
            stop_times['TS_tuned'][ei, si] = r['stopping_time']
            correct['TS_tuned'][ei, si] = r['correct']

    np.savez(os.path.join(OUT, 'mis_1_results.npz'),
             eps=np.array(eps_sweep),
             seeds=np.array(seeds),
             H_eps=np.array(H_eps_vals),
             comp_size=np.array(comp_size),
             eps_critical=np.array([eps_critical[i] for i in sorted(eps_critical)]),
             **{f'{n}_stop': stop_times[n] for n in algo_names},
             **{f'{n}_correct': correct[n].astype(int) for n in algo_names})

    # --- Plot 3-panel figure -------------------------------------------------
    fig, axes = plt.subplots(3, 1, figsize=(9, 11), sharex=True)
    log_eps = np.log10(eps_sweep)

    # Panel A: stopping time
    panel_a_styles = {
        'TS_tuned': {'color': '#d62728', 'marker': 's', 'ls': '-'},
        'TS_rho1':  {'color': '#ff9896', 'marker': 'o', 'ls': '--'},
        'Basic':    {'color': '#2ca02c', 'marker': '^', 'ls': ':'},
    }
    for name, label in [('TS_tuned', 'TS (rho* tuned)'),
                        ('TS_rho1', 'TS (rho=1, fixed)'),
                        ('Basic', 'Basic TS (no graph)')]:
        plotting.plot_with_ci(axes[0], log_eps, stop_times[name], label=label,
                              **panel_a_styles[name])
    # vertical lines at each eps_i^*
    for i, v in eps_critical.items():
        if np.isfinite(v) and v > 0:
            axes[0].axvline(np.log10(v), color='k', alpha=0.3, ls=':',
                            linewidth=1)
    # horizontal asymptote
    a_star = int(np.argmax(mu))
    gap2_nonstar = [(mu[a_star] - mu[i]) ** 2
                    for i in range(K) if i != a_star]
    asym = (1.0 / min(gap2_nonstar)) * np.log(1 / delta)
    axes[0].axhline(asym, color='black', ls='--', alpha=0.5,
                    label=f'max 1/Δ² · log(1/δ) ≈ {asym:.0f}')
    axes[0].set_ylabel('stopping time')
    axes[0].set_yscale('log')
    axes[0].set_title('Panel A: stopping time vs log10(ε)')
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3, which='both')

    # Panel B: H_eps (analytical)
    axes[1].step(log_eps, H_eps_vals, where='mid', color='tab:blue')
    sum_limit = sum(1.0 / g for g in gap2_nonstar)
    max_limit = 1.0 / min(gap2_nonstar)
    axes[1].axhline(sum_limit, color='gray', ls='--', alpha=0.7,
                    label=f'sum 1/Δ² = {sum_limit:.1f}')
    axes[1].axhline(max_limit, color='black', ls='--', alpha=0.7,
                    label=f'max 1/Δ² = {max_limit:.1f}')
    axes[1].set_ylabel('H_ε')
    axes[1].set_title('Panel B: analytical H_ε')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    # Panel C: |competitive set|
    axes[2].step(log_eps, comp_size, where='mid', color='tab:green')
    axes[2].set_xlabel('log10(ε)')
    axes[2].set_ylabel('|competitive set|')
    axes[2].set_title('Panel C: competitive-set size')
    axes[2].grid(alpha=0.3)

    fig.tight_layout()
    out_png = os.path.join(OUT, 'mis_1.png')
    fig.savefig(out_png, dpi=150, bbox_inches='tight')
    print(f"Saved {out_png}")

    # Acceptance
    med_tuned = np.median(stop_times['TS_tuned'], axis=1)
    monotone = all(med_tuned[i] <= med_tuned[i - 1] * 1.2
                   for i in range(1, len(med_tuned)))
    ratio_rho = np.median(stop_times['TS_rho1'][-1]) / max(
        np.median(stop_times['TS_tuned'][-1]), 1.0)
    asym_factor = np.median(stop_times['TS_tuned'][-1]) / max(asym, 1.0)
    print("\nAcceptance:")
    print(f"  [{'x' if monotone else ' '}] TS_tuned stopping time weakly decreasing in eps")
    print(f"  [{'x' if asym_factor <= 3 else ' '}] TS_tuned asymptote within 3x of max-gap limit "
          f"(got x{asym_factor:.2f})")
    print(f"  [{'x' if ratio_rho >= 2 else ' '}] rho-tuning matters: rho=1 / rho* >= 2  "
          f"(got x{ratio_rho:.2f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
