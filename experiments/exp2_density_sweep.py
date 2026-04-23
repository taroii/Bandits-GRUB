"""Experiment 2 - Density sweep: graph-smooth vs graph-feedback.

Validates that H_GF << H_graph on dense graphs and that the three algorithms
converge on sparse graphs.
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', type=int, default=15)
    parser.add_argument('--n-jobs', type=int, default=1)
    parser.add_argument('--quick', action='store_true')
    args = parser.parse_args()

    delta = 1e-3
    if args.quick:
        ps = [0.1, 0.5, 1.0]
        seeds = list(range(max(args.seeds, 3)))
        n = 25
    else:
        ps = [0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
        seeds = list(range(args.seeds))
        n = 50

    algo_names = ['ThompsonSampling', 'GraphFeedbackTS', 'BasicThompsonSampling']
    stop_times = {name: np.zeros((len(ps), len(seeds))) for name in algo_names}
    correct = {name: np.zeros((len(ps), len(seeds)), dtype=bool) for name in algo_names}
    H_vals = {'H_classical': [], 'H_graph': [], 'H_GF': []}

    for pi, p in enumerate(ps):
        mu, A, D = instances.erdos_renyi(n=n, p=p, gap=0.3, seed=pi)
        H_vals['H_classical'].append(hardness.classical_hardness(mu))
        H_vals['H_graph'].append(hardness.graph_hardness(mu, A, D, rho=1.0))
        H_vals['H_GF'].append(hardness.graph_feedback_hardness(mu, A))
        print(f"[exp2] p={p}: H_classical={H_vals['H_classical'][-1]:.1f}, "
              f"H_graph={H_vals['H_graph'][-1]:.1f}, "
              f"H_GF={H_vals['H_GF'][-1]:.1f}", flush=True)

        factories = {
            'ThompsonSampling': lambda D=D, A=A, mu=mu: graph_algo.ThompsonSampling(
                D, A, mu, rho_lap=1.0, delta=delta, q=0.01),
            'GraphFeedbackTS': lambda D=D, A=A, mu=mu: graph_algo.GraphFeedbackTS(
                D, A, mu, delta=delta, q=0.01),
            'BasicThompsonSampling': lambda mu=mu: graph_algo.BasicThompsonSampling(
                mu, delta=delta, q=0.01),
        }
        for name, fac in factories.items():
            print(f"  algo={name} ...", end='', flush=True)
            t0 = time.time()
            runs = runners.run_many(fac, seeds, n_jobs=args.n_jobs,
                                    max_steps=500_000)
            print(f" done in {time.time()-t0:.1f}s (t_med="
                  f"{np.median([r['stopping_time'] for r in runs])})", flush=True)
            for si, r in enumerate(runs):
                stop_times[name][pi, si] = r['stopping_time']
                correct[name][pi, si] = r['correct']

    np.savez(os.path.join(OUT, 'exp2_results.npz'),
             ps=np.array(ps), seeds=np.array(seeds),
             **{f'{n}_stop': stop_times[n] for n in algo_names},
             **{f'{n}_correct': correct[n].astype(int) for n in algo_names},
             **{k: np.array(v) for k, v in H_vals.items()})

    fig, ax = plt.subplots(figsize=(8, 5))
    for name in algo_names:
        style = plotting.style_for(name)
        plotting.plot_with_ci(ax, ps, stop_times[name], label=name, **style)
    log_delta = np.log(1.0 / delta)
    ax.plot(ps, log_delta * np.array(H_vals['H_graph']),
            ':', color='tab:red', alpha=0.7, label='H_graph · log(1/δ)')
    ax.plot(ps, log_delta * np.array(H_vals['H_GF']),
            ':', color='tab:purple', alpha=0.7, label='H_GF · log(1/δ)')
    ax.plot(ps, log_delta * np.array(H_vals['H_classical']),
            ':', color='gray', alpha=0.7, label='H_classical · log(1/δ)')
    ax.set_xlabel('edge probability p')
    ax.set_ylabel('stopping time (log scale)')
    ax.set_yscale('log')
    ax.set_title('Density sweep on Erdos-Renyi graphs')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which='both')
    fig.tight_layout()
    out_png = os.path.join(OUT, 'exp2_density_sweep.png')
    fig.savefig(out_png, dpi=150, bbox_inches='tight')
    print(f"Saved {out_png}")

    # Acceptance
    med = {name: np.median(stop_times[name], axis=1) for name in algo_names}
    ratio_p1 = med['ThompsonSampling'][-1] / max(med['GraphFeedbackTS'][-1], 1.0)
    ratio_sparse = max(med['ThompsonSampling'][0], med['GraphFeedbackTS'][0],
                       med['BasicThompsonSampling'][0]) \
        / max(min(med['ThompsonSampling'][0], med['GraphFeedbackTS'][0],
                  med['BasicThompsonSampling'][0]), 1.0)
    print("\nAcceptance:")
    print(f"  [{'x' if ratio_p1 >= 5 else ' '}] p=1: TS / GFTS >= 5x  (got {ratio_p1:.1f})")
    print(f"  [{'x' if ratio_sparse <= 2 else ' '}] p sparse: max/min <= 2  (got {ratio_sparse:.2f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
