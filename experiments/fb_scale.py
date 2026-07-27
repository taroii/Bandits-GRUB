"""Task 4a -- graph-feedback sweep at larger n (Reviewer 4Y4U W3, "limited scale").

Purpose: measure how T / H_GF varies with n.  Theorem 12 bounds the stopping
time by 186 C(T) L2(T) * H_GF, and Algorithm 2's greedy max-cover pull rule is
an O(log n) approximation to the covering LP that defines H_GF.

We report T / H_GF and also T / (H_GF * L2(T)), since
L2(T) = log(12 K^2 T^2 / delta) itself grows with n independently of the
covering factor.  All four (stop, pull) combinations are run; note that
TS+cover and UCB+cover share one pull rule and TS+width / UCB+width share the
other, which is what makes the pull rule separable from the stopping rule.

  python experiments/fb_scale.py --ns 20 50 100 --p 0.2 --seeds 5
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

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs')
os.makedirs(OUT, exist_ok=True)

COMBOS = [('ts', 'cover'), ('ucb', 'cover'), ('ts', 'width'), ('ucb', 'width')]
LABEL = {('ts', 'cover'): 'TS+cover', ('ucb', 'cover'): 'UCB+cover',
         ('ts', 'width'): 'TS+width', ('ucb', 'width'): 'UCB+width'}


def er_instance(n, p, gap, seed):
    G = nx.erdos_renyi_graph(n, p, seed=seed)
    A = nx.to_numpy_array(G, nodelist=range(n))
    mu = np.full(n, 1.0 - gap)
    mu[0] = 1.0
    return mu, A, np.diag(A.sum(axis=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ns', type=int, nargs='+', default=[20, 50, 100])
    ap.add_argument('--p', type=float, default=0.2)
    ap.add_argument('--gap', type=float, default=0.3)
    ap.add_argument('--seeds', type=int, default=5)
    ap.add_argument('--q', type=float, default=0.1)
    ap.add_argument('--max-steps', type=int, default=5_000_000)
    ap.add_argument('--fresh', action='store_true')
    args = ap.parse_args()

    delta = 1e-3
    seeds = list(range(args.seeds))
    out_npz = os.path.join(OUT, f'fb_scale_p{args.p:g}_results.npz')
    n_n, n_c = len(args.ns), len(COMBOS)

    stop = np.full((n_n, n_c, len(seeds)), np.nan)
    cor = np.zeros((n_n, n_c, len(seeds)), dtype=bool)
    H_GF = np.full(n_n, np.nan)
    H_cls = np.full(n_n, np.nan)
    done = np.zeros(n_n, dtype=bool)

    def save():
        tmp = out_npz + '.tmp.npz'
        np.savez(tmp, ns=np.array(args.ns), seeds=np.array(seeds),
                 combos=np.array([LABEL[c] for c in COMBOS]),
                 stop=stop, correct=cor.astype(int), H_GF=H_GF,
                 H_classical=H_cls, done=done.astype(int),
                 p=float(args.p), gap=float(args.gap), delta=delta,
                 q=args.q, max_steps=int(args.max_steps))
        os.replace(tmp, out_npz)

    if os.path.exists(out_npz) and not args.fresh:
        try:
            z = np.load(out_npz, allow_pickle=False)
            if (list(z['ns'].tolist()) == list(args.ns)
                    and list(z['seeds'].tolist()) == seeds
                    and float(z['p']) == args.p):
                stop, cor = z['stop'], z['correct'].astype(bool)
                H_GF, H_cls = z['H_GF'], z['H_classical']
                done = z['done'].astype(bool)
                print(f"[resume] {done.sum()}/{n_n}", flush=True)
        except Exception as e:
            print(f"[resume] failed: {e}", flush=True)

    print(f"[fb_scale] p={args.p} gap={args.gap} seeds={len(seeds)}",
          flush=True)

    for ni, n in enumerate(args.ns):
        mu0, A0, D0 = er_instance(n, args.p, args.gap, 0)
        if np.isnan(H_GF[ni]):
            H_GF[ni] = hardness.graph_feedback_hardness(mu0, A0)
            H_cls[ni] = hardness.classical_hardness(mu0)
        ncomp = nx.number_connected_components(nx.from_numpy_array(A0))
        print(f"\n=== n={n}  H_GF={H_GF[ni]:.1f}  H_cls={H_cls[ni]:.1f}  "
              f"components={ncomp} ===", flush=True)
        if not done[ni]:
            for ci, (st, pl) in enumerate(COMBOS):
                t0 = time.time()
                ts_, c_ = [], []
                for s in seeds:
                    mu, A, D = er_instance(n, args.p, args.gap, s)

                    def f(mu=mu, A=A, D=D, st=st, pl=pl):
                        return FeedbackGraphBAI(D=D, A=A, mu=mu, delta=delta,
                                                q=args.q, estimator='emp',
                                                stop=st, pull=pl)
                    r = runners.run_algorithm(f, s, max_steps=args.max_steps,
                                              record_elimination=False)
                    ts_.append(r['stopping_time'])
                    c_.append(r['correct'])
                stop[ni, ci] = ts_
                cor[ni, ci] = c_
                save()
                print(f"  {LABEL[(st,pl)]:10s} t_med={np.median(ts_):>10,.0f}  "
                      f"correct={np.mean(c_):.0%}  ({time.time()-t0:.0f}s)",
                      flush=True)
            done[ni] = True
            save()
        else:
            for ci, c in enumerate(COMBOS):
                print(f"  {LABEL[c]:10s} t_med="
                      f"{np.nanmedian(stop[ni,ci]):>10,.0f}  [resumed]",
                      flush=True)

    # ---- summary ------------------------------------------------------
    print("\n" + "=" * 88)
    print("T / H_GF vs n, alongside log n for reference")
    print("=" * 88)
    for ci, c in enumerate(COMBOS):
        print(f"\n  {LABEL[c]}")
        print(f"{'n':>6} {'T_med':>11} {'H_GF':>9} {'T/H_GF':>10} "
              f"{'L2(T)':>8} {'T/(H_GF L2)':>12} {'log n':>7}")
        for ni, n in enumerate(args.ns):
            T = float(np.nanmedian(stop[ni, ci]))
            L2 = np.log(12.0 * n ** 2 * max(T, 1) ** 2 / delta)
            print(f"{n:>6d} {T:>11,.0f} {H_GF[ni]:>9.1f} "
                  f"{T/H_GF[ni]:>10.2f} {L2:>8.2f} "
                  f"{T/(H_GF[ni]*L2):>12.3f} {np.log(n):>7.2f}")
        r = [float(np.nanmedian(stop[ni, ci])) / H_GF[ni]
             for ni in range(n_n)]
        rl = [float(np.nanmedian(stop[ni, ci]))
              / (H_GF[ni] * np.log(12.0*args.ns[ni]**2
                                   * max(np.nanmedian(stop[ni, ci]), 1)**2/delta))
              for ni in range(n_n)]
        print(f"    T/H_GF      n={args.ns[0]}->{args.ns[-1]}: "
              f"{r[0]:.2f} -> {r[-1]:.2f}  "
              f"(x{r[-1]/max(r[0],1e-9):.2f}); log n grows x"
              f"{np.log(args.ns[-1])/np.log(args.ns[0]):.2f}")
        print(f"    T/(H_GF*L2) n={args.ns[0]}->{args.ns[-1]}: "
              f"{rl[0]:.3f} -> {rl[-1]:.3f}  "
              f"(x{rl[-1]/max(rl[0],1e-9):.2f})")
    print(f"\nSaved {out_npz}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
