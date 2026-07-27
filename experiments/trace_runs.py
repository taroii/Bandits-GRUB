"""Task 1 -- produce instrumented runs and a Q7-oriented summary.

Reviewer 4Y4U's Q7 asks *what graph properties drive the reduction*.  This
runner answers it from traces: for each arm it records the effective sample
size t_eff,i actually achieved per direct pull (the "pooling factor"), which
is the mechanism by which the graph buys sample complexity, and cross-tabs
it against the arm's influence factor J(i,G) and gap.

Writes one .npz per (instance, algo, seed) under outputs/traces/, each
re-aggregatable without rerunning.

  python experiments/trace_runs.py --instance chain --seeds 3
  python experiments/trace_runs.py --instance er20 --seeds 3
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.utils import instances, hardness, runners, tracing  # noqa: E402
import graph_algo  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs')
TRACE_DIR = os.path.join(OUT, 'traces')


def build_instance(name, args):
    """Return (tag, mu, A, D, rho, algos) for the named instance."""
    if name == 'chain':
        mu, A, D = instances.clustered_chain(args.K, C=2, gap_step=0.3)
        rho = args.rho
        algos = ['TS-Explore', 'Basic TS', 'KL-LUCB', 'GRUB']
        tag = f'chain_K{args.K}_rho{rho:g}'
    elif name == 'sbm31':
        mu, A, D = instances.sbm_phase_transition_connected()
        rho = args.rho
        algos = ['TS-Explore', 'Basic TS', 'KL-LUCB', 'GRUB']
        tag = f'sbm31_rho{rho:g}'
    elif name == 'er20':
        mu, A, D = instances.erdos_renyi(n=args.K, p=args.p, gap=0.3)
        rho = 0.0
        algos = ['TS-Explore-GF', 'UCB+cover', 'TS+width', 'UCB-N',
                 'TS-Explore-GF(pair)']
        tag = f'er{args.K}_p{args.p:g}'
    else:
        raise ValueError(name)
    return tag, mu, A, D, rho, algos


def make_factory(name, D, A, mu, delta, q, rho):
    if name == 'TS-Explore':
        return lambda: graph_algo.ThompsonSampling(
            D=D, A=A, mu=mu, rho_lap=rho, delta=delta, q=q)
    if name == 'Basic TS':
        return lambda: graph_algo.BasicThompsonSampling(
            mu=mu, delta=delta, q=q)
    if name == 'KL-LUCB':
        return lambda: graph_algo.KL_LUCB(mu=mu, delta=delta)
    if name == 'GRUB':
        return lambda: graph_algo.MaxVarianceArmAlgo(
            D=D, A=A, mu=mu, rho_lap=rho, delta=delta)
    if name == 'TS-Explore-GF':
        return lambda: graph_algo.GraphFeedbackTS(
            D=D, A=A, mu=mu, delta=delta, q=q, pull_scope='all')
    if name == 'TS-Explore-GF(pair)':
        return lambda: graph_algo.GraphFeedbackTS(
            D=D, A=A, mu=mu, delta=delta, q=q, pull_scope='pair')
    if name == 'TS+width':
        return lambda: graph_algo.GraphFeedbackTSWidth(
            D=D, A=A, mu=mu, delta=delta, q=q)
    if name == 'UCB-N':
        return lambda: graph_algo.UCB_N(D=D, A=A, mu=mu, delta=delta)
    if name == 'UCB+cover':
        return lambda: graph_algo.UCBNCover(D=D, A=A, mu=mu, delta=delta)
    raise ValueError(name)


def summarize_trace(tr):
    """Per-run scalars that speak to Q7.

    ``pool`` is the pooling factor t_eff,i / t_i: how many effective samples
    an arm ended up with per direct pull it paid for.  >1 means the graph
    delivered information the arm never bought, which is the mechanism
    behind the reduction.  It is only meaningful for arms that were pulled
    at least once, so arms with t_i = 0 are excluded rather than being
    divided by zero (for the feedback algorithms most arms are never the
    *chosen action*, so including them produced meaningless 1e14 values).

    ``obs_per_pull`` = sum_i N_i^fb / T is the corresponding quantity for
    the feedback setting: the mean number of observations harvested per
    pull, i.e. the realized information multiplier of the graph.
    """
    teff_end = tr['teff'][-1]
    pulls_end = tr['pulls'][-1]
    nfb_end = tr['n_fb'][-1]
    T = max(float(tr['stopping_time']), 1.0)
    pulled = pulls_end >= 1
    if pulled.any():
        pool = teff_end[pulled] / pulls_end[pulled]
        pool_med, pool_max = float(np.median(pool)), float(pool.max())
    else:
        pool_med = pool_max = float('nan')
    a_star = int(tr['a_star'])
    pool_best = (float(teff_end[a_star] / pulls_end[a_star])
                 if pulls_end[a_star] >= 1 else float('nan'))
    # The competitive-set criteria are defined for the graph-smooth setting
    # (they involve rho and eps); at rho = 0 they degenerate, so report NaN
    # rather than a misleading "all arms competitive".
    has_rho = float(tr['rho']) > 0
    return dict(
        T=T,
        pool_med=pool_med,
        pool_max=pool_max,
        pool_best=pool_best,
        n_pulled=int(pulled.sum()),
        teff_sum_over_T=float(teff_end.sum() / T),
        obs_per_pull=float(nfb_end.sum() / T),
        frac_pulls_on_best=float(pulls_end[a_star] / max(pulls_end.sum(), 1)),
        n_compH=int(tr['comp_H'][-1].sum()) if has_rho else -1,
        n_compH_def6=int(tr['comp_H_def6'][-1].sum()) if has_rho else -1,
        n_compH_thaker=int(tr['comp_H_thaker'][-1].sum()) if has_rho else -1,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--instance', choices=['chain', 'sbm31', 'er20'],
                    default='chain')
    ap.add_argument('--K', type=int, default=50,
                    help="arms (chain / er20 'n')")
    ap.add_argument('--p', type=float, default=0.2, help="er20 density")
    ap.add_argument('--rho', type=float, default=100.0)
    ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--q', type=float, default=0.1)
    ap.add_argument('--max-steps', type=int, default=10_000_000)
    ap.add_argument('--every', type=int, default=200,
                    help="snapshot interval in rounds")
    ap.add_argument('--algos', type=str, nargs='+', default=None)
    ap.add_argument('--skip-existing', action='store_true', default=True)
    args = ap.parse_args()

    delta = 1e-3
    os.makedirs(TRACE_DIR, exist_ok=True)
    tag, mu, A, D, rho, algos = build_instance(args.instance, args)
    if args.algos:
        algos = args.algos
    L = D - A
    eps = float(np.sqrt(max(mu @ L @ mu, 0.0)))
    a_star = int(np.argmax(mu))
    gaps = mu[a_star] - mu
    R = (hardness.influence_factors(A, D) if A.sum() > 0
         else np.full((len(mu), len(mu)), np.inf))
    J = R[:, a_star]

    print(f"[trace_runs] {tag}: K={len(mu)} rho={rho:g} eps={eps:.4f} "
          f"a*={a_star} Delta_min={gaps[gaps>0].min():.3f}", flush=True)
    print(f"  H_classical={hardness.classical_hardness(mu):.1f}", flush=True)
    if A.sum() > 0:
        print(f"  H_graph(rho={rho:g})="
              f"{hardness.graph_hardness(mu, A, D, rho=max(rho,1e-9)):.1f}  "
              f"H_GF={hardness.graph_feedback_hardness(mu, A):.1f}", flush=True)
        finite_J = J[np.isfinite(J)]
        print(f"  J(i,G) to a*: min={finite_J.min():.3f} "
              f"med={np.median(finite_J):.3f} max={finite_J.max():.3f}",
              flush=True)

    spec_base = dict(means=mu, Adj=A, Degree=D, rho=rho, eps=eps,
                     delta=delta, q=args.q, every=args.every)

    print(f"\n{'algo':22s} {'seed':>4s} {'T':>10s} {'pool_med':>9s} "
          f"{'pool_a*':>8s} {'obs/pull':>9s} {'|H|':>4s} {'cap':>4s}",
          flush=True)
    rows = {}
    for name in algos:
        fac = make_factory(name, D, A, mu, delta, args.q, rho)
        rows[name] = []
        for seed in range(args.seeds):
            path = os.path.join(TRACE_DIR, f'{tag}_{name}_seed{seed}.npz')
            if args.skip_existing and os.path.exists(path):
                tr = dict(np.load(path, allow_pickle=False))
                s = summarize_trace(tr)
                rows[name].append(s)
                print(f"{name:22s} {seed:>4d} {s['T']:>10,.0f} "
                      f"{s['pool_med']:>9.2f} {s['pool_best']:>8.2f} "
                      f"{s['obs_per_pull']:>9.2f} {s['n_compH']:>4d} "
                      f"{'  --':>4s}  [cached]", flush=True)
                continue
            t0 = time.time()
            r = runners.run_algorithm(fac, seed, max_steps=args.max_steps,
                                      record_elimination=False,
                                      trace_spec=spec_base)
            tr = r['trace']
            tracing.save_trace(tr, path)
            s = summarize_trace(tr)
            rows[name].append(s)
            print(f"{name:22s} {seed:>4d} {s['T']:>10,.0f} "
                  f"{s['pool_med']:>9.2f} {s['pool_best']:>8.2f} "
                  f"{s['obs_per_pull']:>9.2f} {s['n_compH']:>4d} "
                  f"{'Y' if tr['capped'] else 'n':>4s}  "
                  f"({time.time()-t0:.0f}s)", flush=True)

    print(f"\n# medians over {args.seeds} seeds")
    for name, rs in rows.items():
        if not rs:
            continue
        print(f"  {name:22s} T={np.median([r['T'] for r in rs]):>10,.0f}  "
              f"pool_med={np.median([r['pool_med'] for r in rs]):.2f}  "
              f"pool_max={np.median([r['pool_max'] for r in rs]):.2f}  "
              f"obs/pull={np.median([r['obs_per_pull'] for r in rs]):.2f}")
    print(f"\nTraces in {TRACE_DIR}")
    print("Competitive-set sizes at stop (3 criteria, last snapshot):")
    for name, rs in rows.items():
        if rs:
            print(f"  {name:22s} plan={rs[0]['n_compH']:>3d}  "
                  f"def6={rs[0]['n_compH_def6']:>3d}  "
                  f"thaker={rs[0]['n_compH_thaker']:>3d}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
