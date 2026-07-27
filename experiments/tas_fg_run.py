"""Task 6 -- TaS-FG against our graph-feedback algorithms, persisted to .npz.

Places TaS-FG (Russo, Song & Pacchiano, AISTATS 2025) on the same axis as
TS-Explore-GF and UCB+cover on the Erdos-Renyi feedback instance. Records
stopping time, correctness, cap-hits, and per-round wall-clock (which differs
by orders of magnitude between the methods, so it is logged alongside).

Checkpointed per (density, algorithm) cell.

  python experiments/tas_fg_run.py --ps 0.2 0.6 --seeds 5
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import networkx as nx

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.utils import hardness, runners  # noqa: E402
from experiments.utils.feedback_reg import FeedbackGraphBAI  # noqa: E402
from experiments.utils.tas_fg import TaSFG, beta_threshold  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs')
os.makedirs(OUT, exist_ok=True)

ALGOS = ['TaS-FG', 'TS-Explore-GF', 'UCB+cover', 'TS+width', 'UCB-N']


def er_instance(n, p, gap, seed):
    G = nx.erdos_renyi_graph(n, p, seed=seed)
    A = nx.to_numpy_array(G, nodelist=range(n))
    mu = np.full(n, 1.0 - gap)
    mu[0] = 1.0
    return mu, A, np.diag(A.sum(axis=1))


def factory(name, D, A, mu, delta, q, resolve_every):
    if name == 'TaS-FG':
        return lambda: TaSFG(D=D, A=A, mu=mu, delta=delta,
                             resolve_every=resolve_every)
    spec = {'TS-Explore-GF': ('ts', 'cover'), 'UCB+cover': ('ucb', 'cover'),
            'TS+width': ('ts', 'width'), 'UCB-N': ('ucb', 'width')}[name]
    return lambda: FeedbackGraphBAI(D=D, A=A, mu=mu, delta=delta, q=q,
                                    estimator='emp', stop=spec[0],
                                    pull=spec[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=20)
    ap.add_argument('--ps', type=float, nargs='+', default=[0.2, 0.4, 0.6])
    ap.add_argument('--gap', type=float, default=0.3)
    ap.add_argument('--seeds', type=int, default=5)
    ap.add_argument('--q', type=float, default=0.1)
    ap.add_argument('--resolve-every', type=int, default=25)
    ap.add_argument('--max-steps', type=int, default=2_000_000)
    ap.add_argument('--algos', type=str, nargs='+', default=ALGOS)
    ap.add_argument('--fresh', action='store_true')
    args = ap.parse_args()

    delta = 1e-3
    seeds = list(range(args.seeds))
    out_npz = os.path.join(OUT, f'tas_fg_n{args.n}_results.npz')
    n_p, n_a = len(args.ps), len(ALGOS)

    stop = np.full((n_p, n_a, len(seeds)), np.nan)
    cor = np.zeros((n_p, n_a, len(seeds)), dtype=bool)
    cap = np.zeros((n_p, n_a, len(seeds)), dtype=bool)
    secs = np.full((n_p, n_a), np.nan)
    H_GF = np.full(n_p, np.nan)
    done = np.zeros((n_p, n_a), dtype=bool)

    def save():
        tmp = out_npz + '.tmp.npz'
        np.savez(tmp, ps=np.array(args.ps), seeds=np.array(seeds),
                 algos=np.array(ALGOS), stop=stop, correct=cor.astype(int),
                 capped=cap.astype(int), seconds=secs, H_GF=H_GF,
                 done=done.astype(int), n=int(args.n), gap=float(args.gap),
                 delta=delta, q=args.q,
                 resolve_every=int(args.resolve_every),
                 max_steps=int(args.max_steps))
        os.replace(tmp, out_npz)

    if os.path.exists(out_npz) and not args.fresh:
        try:
            z = np.load(out_npz, allow_pickle=False)
            if (list(z['ps'].tolist()) == list(args.ps)
                    and list(z['seeds'].tolist()) == seeds
                    and int(z['n']) == args.n):
                stop, cor = z['stop'], z['correct'].astype(bool)
                cap, secs = z['capped'].astype(bool), z['seconds']
                H_GF, done = z['H_GF'], z['done'].astype(bool)
                print(f"[resume] {int(done.sum())}/{n_p*n_a} cells", flush=True)
        except Exception as e:
            print(f"[resume] failed: {e}", flush=True)

    print(f"[tas_fg] n={args.n} gap={args.gap} seeds={len(seeds)} "
          f"resolve_every={args.resolve_every}  "
          f"beta(1e4)={beta_threshold(1e4, delta, args.n):.2f}", flush=True)

    for pi, p in enumerate(args.ps):
        mu0, A0, _ = er_instance(args.n, p, args.gap, 0)
        if np.isnan(H_GF[pi]):
            H_GF[pi] = hardness.graph_feedback_hardness(mu0, A0)
        print(f"\n=== p={p:g}  H_GF={H_GF[pi]:.1f} ===", flush=True)
        for ai, name in enumerate(ALGOS):
            if name not in args.algos:
                continue
            if done[pi, ai]:
                print(f"  {name:14s} t_med={np.nanmedian(stop[pi,ai]):>10,.0f}"
                      f"  [resumed]", flush=True)
                continue
            t0 = time.time()
            ts_, c_, k_ = [], [], []
            for s in seeds:
                mu, A, D = er_instance(args.n, p, args.gap, s)
                r = runners.run_algorithm(
                    factory(name, D, A, mu, delta, args.q,
                            args.resolve_every), s,
                    max_steps=args.max_steps, record_elimination=False)
                ts_.append(r['stopping_time'])
                c_.append(r['correct'])
                k_.append(not r['converged_flag'])
            stop[pi, ai], cor[pi, ai], cap[pi, ai] = ts_, c_, k_
            secs[pi, ai] = time.time() - t0
            done[pi, ai] = True
            save()
            per_round = secs[pi, ai] / max(np.sum(ts_), 1) * 1e3
            print(f"  {name:14s} t_med={np.median(ts_):>10,.0f}  "
                  f"T/H_GF={np.median(ts_)/H_GF[pi]:>7.1f}  "
                  f"correct={np.mean(c_):.0%}  cap={int(np.sum(k_))}/"
                  f"{len(seeds)}  {per_round:.3f} ms/round  "
                  f"({secs[pi,ai]:.0f}s)", flush=True)

    save()
    print("\n# Summary")
    print(f"{'p':>6} {'H_GF':>7} " + " ".join(f"{a:>15}" for a in ALGOS))
    for pi, p in enumerate(args.ps):
        row = f"{p:>6g} {H_GF[pi]:>7.1f} "
        for ai in range(n_a):
            v = np.nanmedian(stop[pi, ai])
            mark = '*' if cap[pi, ai].any() else ''
            row += f"{(f'{v:,.0f}' + mark) if v == v else '--':>16}"
        print(row)
    print("\n# T / H_GF")
    print(f"{'p':>6} " + " ".join(f"{a:>15}" for a in ALGOS))
    for pi, p in enumerate(args.ps):
        row = f"{p:>6g} "
        for ai in range(n_a):
            v = np.nanmedian(stop[pi, ai]) / H_GF[pi]
            row += f"{(f'{v:.1f}' if v == v else '--'):>16}"
        print(row)
    print("\n# ms per round (wall-clock, single-threaded)")
    print(f"{'p':>6} " + " ".join(f"{a:>15}" for a in ALGOS))
    for pi, p in enumerate(args.ps):
        row = f"{p:>6g} "
        for ai in range(n_a):
            tot = np.nansum(stop[pi, ai])
            v = secs[pi, ai] / tot * 1e3 if tot > 0 else np.nan
            row += f"{(f'{v:.3f}' if v == v else '--'):>16}"
        print(row)
    print("\n  * = at least one seed hit the pull cap")
    print(f"\nSaved {out_npz}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
