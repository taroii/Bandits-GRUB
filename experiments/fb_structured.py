"""fb_structured -- graph-feedback validation on canonical graph families.

Companion to ``fb_1.py``: that script sweeps Erdos-Renyi density;
this one fixes n=20 and uniform suboptimality gap Delta=0.3 and
varies the graph family (clique, star, k-regular, Barabasi-Albert).
The point is to validate the combinatorial bounds on H_GF in
Corollary 5.2 against the empirical stopping times of the
graph-feedback algorithms.

Algorithms (all with delta=1e-3):
  * TS-Explore-GF   - proposed graph-feedback Thompson sampler.
  * UCB+cover       - UCB-LCB elimination using the same cover-pair
                      pull rule.
  * KL-LUCB         - non-graph fixed-confidence reference.

Saves results to experiments/outputs/fb_structured_results.npz with
H_GF (analytical), bar-chi (clique-cover), 2-packing rho, and
min-degree per instance for direct comparison to Corollary 5.2.

Crash-proof: strictly sequential (no multiprocessing), per-cell
try/except, checkpoint after every cell, max-steps cap.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import traceback

import numpy as np
import networkx as nx

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.utils import hardness, runners  # noqa: E402
import graph_algo  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs')
os.makedirs(OUT, exist_ok=True)


ALGOS = ['TS-Explore-GF', 'UCB+cover', 'KL-LUCB']


def _mats_from_graph(G, n):
    A = np.zeros((n, n))
    for i, j in G.edges():
        A[i, j] = 1.0
        A[j, i] = 1.0
    D = np.diag(A.sum(axis=1))
    return A, D


def build_instance(family, n, gap, seed):
    """Return (mu, A, D) with arm 0 having mean 1.0 and the rest 1-gap."""
    if family == 'clique':
        G = nx.complete_graph(n)
    elif family == 'star':
        # Star on n vertices: center index 0, leaves 1..n-1.
        G = nx.star_graph(n - 1)
    elif family == 'k4-regular':
        # 4-regular on n vertices; n*k must be even (n=20, k=4 -> ok).
        G = nx.random_regular_graph(4, n, seed=seed)
    elif family == 'BA':
        G = nx.barabasi_albert_graph(n, 2, seed=seed)
    else:
        raise ValueError(f"unknown family: {family}")
    A, D = _mats_from_graph(G, n)
    mu = np.full(n, 1.0 - gap)
    mu[0] = 1.0
    return mu, A, D


def graph_stats(A, D):
    """Return (n_edges, d_min, bar_chi, two_packing) on the graph."""
    n = A.shape[0]
    n_edges = int(A.sum() / 2)
    d_min = int(np.diag(D).min())
    G = nx.from_numpy_array(A)
    # Clique-cover number = chromatic number of complement.
    Gc = nx.complement(G)
    try:
        bar_chi = int(max(nx.coloring.greedy_color(Gc).values()) + 1)
    except Exception:
        bar_chi = -1
    # 2-packing number: largest set whose closed neighborhoods are
    # pairwise disjoint. Greedy lower bound; small n=20 makes this fast.
    closed = (A + np.eye(n)) > 0
    used = np.zeros(n, dtype=bool)
    packing = []
    for v in range(n):
        if used[v]:
            continue
        packing.append(v)
        used[closed[v]] = True
    return n_edges, d_min, bar_chi, len(packing)


def build_factory(name, A, mu, delta, q):
    if name == 'TS-Explore-GF':
        D = np.diag(A.sum(axis=1))
        return lambda: graph_algo.GraphFeedbackTS(
            D=D, A=A, mu=mu, delta=delta, q=q)
    if name == 'UCB+cover':
        D = np.diag(A.sum(axis=1))
        return lambda: graph_algo.UCBNCover(
            D=D, A=A, mu=mu, delta=delta)
    if name == 'KL-LUCB':
        # Non-graph baseline: only mu matters.
        return lambda: graph_algo.KL_LUCB(mu=mu, delta=delta)
    raise ValueError(name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=20)
    parser.add_argument('--gap', type=float, default=0.3)
    parser.add_argument('--seeds', type=int, default=10)
    parser.add_argument('--q', type=float, default=0.1)
    parser.add_argument('--max-steps', type=int, default=500_000)
    parser.add_argument('--families', type=str, nargs='+',
                        default=['clique', 'star', 'k4-regular', 'BA'])
    parser.add_argument('--fresh', action='store_true')
    args = parser.parse_args()

    families = args.families
    seeds = list(range(args.seeds))
    delta = 1e-3
    out_npz = os.path.join(OUT, 'fb_structured_results.npz')

    n_fam = len(families)
    stop_times = {a: np.full((n_fam, len(seeds)), np.nan) for a in ALGOS}
    correct = {a: np.zeros((n_fam, len(seeds)), dtype=bool) for a in ALGOS}
    H_GF = np.full((n_fam, len(seeds)), np.nan)
    H_classical = np.full(n_fam, np.nan)
    n_edges_arr = np.full((n_fam, len(seeds)), np.nan)
    d_min_arr = np.full((n_fam, len(seeds)), np.nan)
    bar_chi_arr = np.full((n_fam, len(seeds)), np.nan)
    two_packing_arr = np.full((n_fam, len(seeds)), np.nan)
    done = np.zeros((n_fam, len(ALGOS), len(seeds)), dtype=bool)

    def save_checkpoint():
        kwargs = dict(
            families=np.array(families),
            algo_names=np.array(ALGOS),
            seeds=np.array(seeds),
            n=int(args.n),
            gap=float(args.gap),
            delta=delta,
            q=args.q,
            H_GF=H_GF,
            H_classical=H_classical,
            n_edges=n_edges_arr,
            d_min=d_min_arr,
            bar_chi=bar_chi_arr,
            two_packing=two_packing_arr,
            done=done.astype(int),
        )
        for a in ALGOS:
            kwargs[f'{a}_stop'] = stop_times[a]
            kwargs[f'{a}_correct'] = correct[a].astype(int)
        tmp = out_npz + '.tmp.npz'
        np.savez(tmp, **kwargs)
        os.replace(tmp, out_npz)

    if os.path.exists(out_npz) and not args.fresh:
        try:
            z = np.load(out_npz, allow_pickle=False)
            prev_families = list(z['families'].astype(str).tolist())
            prev_algos = list(z['algo_names'].astype(str).tolist())
            prev_seeds = list(z['seeds'].tolist())
            if (prev_families == families
                    and prev_algos == ALGOS
                    and prev_seeds == seeds
                    and int(z['n']) == args.n
                    and abs(float(z['gap']) - args.gap) < 1e-9):
                done = z['done'].astype(bool)
                H_GF = z['H_GF']
                H_classical = z['H_classical']
                n_edges_arr = z['n_edges']
                d_min_arr = z['d_min']
                bar_chi_arr = z['bar_chi']
                two_packing_arr = z['two_packing']
                for a in ALGOS:
                    stop_times[a] = z[f'{a}_stop']
                    correct[a] = z[f'{a}_correct'].astype(bool)
                print(f"[resume] loaded {int(done.sum())}/"
                      f"{n_fam * len(ALGOS) * len(seeds)} cells",
                      flush=True)
            else:
                print("[resume] checkpoint mismatch; starting fresh "
                      "(use --fresh to silence)", flush=True)
        except Exception as e:
            print(f"[resume] failed to load checkpoint: {e}", flush=True)

    for fi, family in enumerate(families):
        print(f"\n=== family={family} ===", flush=True)
        # H_classical is the same across seeds (depends only on mu structure).
        try:
            mu0, A0, D0 = build_instance(family, args.n, args.gap, seed=0)
            H_classical[fi] = hardness.classical_hardness(mu0)
        except Exception as e:
            print(f"  [skip family] failed to build seed=0: {e}", flush=True)
            traceback.print_exc()
            continue

        for si, seed in enumerate(seeds):
            try:
                mu, A, D = build_instance(family, args.n, args.gap, seed=seed)
                H_GF[fi, si] = hardness.graph_feedback_hardness(mu, A)
                ne, dm, bc, tp = graph_stats(A, D)
                n_edges_arr[fi, si] = ne
                d_min_arr[fi, si] = dm
                bar_chi_arr[fi, si] = bc
                two_packing_arr[fi, si] = tp
            except Exception as e:
                print(f"  [skip seed-build] {family} seed={seed}: {e}",
                      flush=True)
                traceback.print_exc()
                continue

            for ai, name in enumerate(ALGOS):
                if done[fi, ai, si]:
                    continue
                fac = build_factory(name, A, mu, delta, args.q)
                t0 = time.time()
                try:
                    out = runners.run_algorithm(
                        fac, seed=seed, max_steps=args.max_steps,
                        record_elimination=False)
                    stop_times[name][fi, si] = out['stopping_time']
                    correct[name][fi, si] = out['correct']
                    done[fi, ai, si] = True
                except Exception as e:
                    print(f"  [skip] {family} {name} seed={seed} failed: {e}",
                          flush=True)
                    traceback.print_exc()
                    stop_times[name][fi, si] = np.nan
                    correct[name][fi, si] = False
                    done[fi, ai, si] = True
                save_checkpoint()
                print(f"  {family:12s} {name:14s} seed={seed} "
                      f"t={stop_times[name][fi, si]:.0f} "
                      f"correct={int(correct[name][fi, si])} "
                      f"({time.time() - t0:.1f}s)",
                      flush=True)

        # Per-family summary across seeds.
        if not np.all(np.isnan(H_GF[fi])):
            print(f"  H_GF(median)={np.nanmedian(H_GF[fi]):.1f}  "
                  f"bar_chi(median)={int(np.nanmedian(bar_chi_arr[fi]))}  "
                  f"2-packing(median)={int(np.nanmedian(two_packing_arr[fi]))}  "
                  f"d_min(median)={int(np.nanmedian(d_min_arr[fi]))}",
                  flush=True)

    print(f"\nSaved {out_npz}")
    print()
    print("# Per-family medians (stopping time vs combinatorial bounds)")
    header = (f"{'family':<12s}  {'H_cls':>7s}  {'H_GF':>7s}  "
              f"{'bar_chi':>7s}  {'2-pack':>6s}  "
              + "  ".join(f"{a + ' med':>16s}" for a in ALGOS))
    print(header)
    print('-' * len(header))
    for fi, family in enumerate(families):
        h_gf = np.nanmedian(H_GF[fi]) if not np.all(np.isnan(H_GF[fi])) else float('nan')
        bc = np.nanmedian(bar_chi_arr[fi]) if not np.all(np.isnan(bar_chi_arr[fi])) else float('nan')
        tp = np.nanmedian(two_packing_arr[fi]) if not np.all(np.isnan(two_packing_arr[fi])) else float('nan')
        row = (f"{family:<12s}  {H_classical[fi]:>7.1f}  {h_gf:>7.1f}  "
               f"{bc:>7.0f}  {tp:>6.0f}  ")
        row += "  ".join(
            f"{np.nanmedian(stop_times[a][fi, :]):>16,.0f}" for a in ALGOS)
        print(row)


if __name__ == "__main__":
    main()
