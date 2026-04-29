"""Experiment 6 - Figure 1 with seed variance.

Replicates the sample_main.py plot on the config.toml SBM instance but
aggregates across 30 seeds so the reader can see the variance of each
UCB-style baseline.  Thompson Sampling does not expose intermediate
eliminations; we render its stopping time as a shaded vertical band.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.utils import instances, runners, plotting  # noqa: E402
import graph_algo  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs')
os.makedirs(OUT, exist_ok=True)


UCB_ALGOS = ['MaxDiffVarAlgo', 'CyclicAlgo', 'OneStepMinSumAlgo',
             'MaxVarianceArmAlgo', 'NoGraphAlgo']
TS_NAME = 'ThompsonSampling'


def interpolate_curve(curve, t_grid):
    """Given a list of (t, remaining) in increasing t, return remaining(t_grid)."""
    ts = np.array([p[0] for p in curve], dtype=float)
    rs = np.array([p[1] for p in curve], dtype=float)
    # Step function: between t_k and t_{k+1}, remaining stays rs[k]
    idx = np.searchsorted(ts, t_grid, side='right') - 1
    idx = np.clip(idx, 0, len(rs) - 1)
    return rs[idx]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', type=int, default=10)
    parser.add_argument('--n-jobs', type=int, default=1)
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--q', type=float, default=0.1)
    parser.add_argument('--max-steps', type=int, default=300_000)
    args = parser.parse_args()

    seeds = list(range(args.seeds if not args.quick else 3))
    # Downscaled from notes.md's K=101 SBM for tractability
    mu, A, D = instances.sbm_standard(
        n_clusters=2, nodes_per_cluster=5, p=0.9, q=0.0,
        best_factor=1.3, seed=0)
    K = len(mu)
    rho_lap = 1.0
    delta = 1e-3
    q = args.q

    def make_factory(cls_name):
        cls = getattr(graph_algo, cls_name)
        if cls_name == TS_NAME:
            return lambda: cls(D, A, mu, rho_lap=rho_lap, delta=delta, q=q)
        # UCB baselines don't take delta in their constructor; AlgoBase
        # hardcodes self.delta=0.0001.  Override post-init so they share
        # delta with TS.
        def _build(cls=cls, cls_name=cls_name):
            inst = (cls(D, A, mu) if cls_name == 'NoGraphAlgo'
                    else cls(D, A, mu, rho_lap=rho_lap))
            inst.delta = delta
            return inst
        return _build

    curves = {}
    stop_times = {}
    for name in UCB_ALGOS + [TS_NAME]:
        fac = make_factory(name)
        print(f"[exp6] algo={name} ...", flush=True)
        t0 = time.time()
        runs = runners.run_many(fac, seeds, n_jobs=args.n_jobs,
                                max_steps=args.max_steps)
        curves[name] = [r['elimination_curve'] for r in runs]
        stop_times[name] = np.array([r['stopping_time'] for r in runs], dtype=float)
        print(f"    t_med={np.median(stop_times[name]):.0f} "
              f"({time.time()-t0:.1f}s)", flush=True)

    np.savez(os.path.join(OUT, 'exp6_results.npz'),
             **{f'{n}_stop': stop_times[n] for n in stop_times})

    # Build time grid up to the max UCB stopping time
    max_t = max(stop_times[n].max() for n in UCB_ALGOS)
    t_grid = np.linspace(0, max_t, 400)

    fig, ax = plt.subplots(figsize=(10, 6))
    for name in UCB_ALGOS:
        interp = np.stack([interpolate_curve(c, t_grid) for c in curves[name]])
        med = np.median(interp, axis=0)
        lo = np.percentile(interp, 25, axis=0)
        hi = np.percentile(interp, 75, axis=0)
        style = plotting.style_for(name)
        ax.plot(t_grid, med, label=name, color=style['color'],
                ls=style['ls'], linewidth=1.8)
        ax.fill_between(t_grid, lo, hi, color=style['color'], alpha=0.15)

    # TS: vertical shaded band
    ts_lo, ts_hi = np.percentile(stop_times[TS_NAME], [25, 75])
    ts_med = np.median(stop_times[TS_NAME])
    ax.axvspan(ts_lo, ts_hi, color='tab:red', alpha=0.15, label='TS IQR')
    ax.axvline(ts_med, color='tab:red', linestyle='--', alpha=0.8,
               label=f'TS median = {ts_med:.0f}')

    ax.set_xlabel('time steps (pulls)')
    ax.set_ylabel('remaining arms')
    ax.set_title(f'Fig 1 with seed variance ({len(seeds)} seeds)')
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_png = os.path.join(OUT, 'exp6_fig1_with_ci.png')
    fig.savefig(out_png, dpi=150, bbox_inches='tight')
    print(f"Saved {out_png}")

    ucb_medians = {n: np.median(stop_times[n]) for n in UCB_ALGOS}
    ts_hi_quantile = np.percentile(stop_times[TS_NAME], 75)
    left_of_all = all(ts_hi_quantile < m for m in ucb_medians.values())
    print(f"\nAcceptance:  [{'x' if left_of_all else ' '}] TS p75 < all UCB medians")
    return 0


if __name__ == "__main__":
    sys.exit(main())
