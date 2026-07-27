"""Task 2 -- combined smoothness + graph feedback (highest-priority new run).

2x2x2 design on an Erdos-Renyi feedback graph:
    estimator {emp, reg} x stop {ts, ucb} x pull {cover, width}
The 'emp' half reproduces the current Figure 2 (right panel); the 'reg' half
is the new Laplacian-regularized-over-side-observations variant.

Prediction under test: the TS advantage comes from the agreement stopping rule
being insensitive to an inflated confidence radius.  With the unbiased
empirical estimator there is no inflation and no TS advantage (UCB+cover wins,
as in current Figure 2).  With the regularized estimator the UCB radius must
carry a sqrt(rho)*eps bias term, so TS should regain ground.  Report either
way -- this is the run that can come back negative.

Instances (`--instance`):
  er_uniform  the current Figure 2 instance: ER G(n,p) on all n nodes,
              mu_0 = 1, everything else 1-Delta.  Not smooth on G: every edge
              incident to arm 0 contributes Delta^2, so eps^2 = deg(0) Delta^2.
  er_smooth   rewards made smooth on G: the n-1 challengers form ER G(n-1,p)
              and all share mean 1-Delta; the best arm is attached by only
              `--n-bridge` edges, so eps^2 = n_bridge * Delta^2 exactly and
              ||mu||_G is genuinely small and *independent of p*.
              This keeps the swept axis (challenger density p) comparable to
              Figure 2 while making the instance smooth.

Realized eps is reported per graph realization in both cases.

  python experiments/fb_smooth.py --instance er_smooth --seeds 5 \
      --rhos 1 10 100 1000
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


def build_instance(kind, n, p, gap, n_bridge, seed):
    if kind == 'er_uniform':
        G = nx.erdos_renyi_graph(n, p, seed=seed)
        A = nx.to_numpy_array(G, nodelist=range(n))
        mu = np.full(n, 1.0 - gap)
        mu[0] = 1.0
    elif kind == 'er_smooth':
        # challengers 1..n-1 form ER(n-1, p); best arm 0 bridges to n_bridge
        Gc = nx.erdos_renyi_graph(n - 1, p, seed=seed)
        A = np.zeros((n, n))
        for i, j in Gc.edges():
            A[i + 1, j + 1] = A[j + 1, i + 1] = 1.0
        rng = np.random.default_rng(seed)
        bridges = rng.choice(np.arange(1, n), size=min(n_bridge, n - 1),
                             replace=False)
        for b in bridges:
            A[0, b] = A[b, 0] = 1.0
        mu = np.full(n, 1.0 - gap)
        mu[0] = 1.0
    else:
        raise ValueError(kind)
    D = np.diag(A.sum(axis=1))
    return mu, A, D


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--instance', choices=['er_uniform', 'er_smooth'],
                    default='er_smooth')
    ap.add_argument('--n', type=int, default=20)
    ap.add_argument('--ps', type=float, nargs='+',
                    default=[0.1, 0.2, 0.4, 0.6, 0.8, 1.0])
    ap.add_argument('--rhos', type=float, nargs='+',
                    default=[1.0, 10.0, 100.0, 1000.0])
    ap.add_argument('--gap', type=float, default=0.3)
    ap.add_argument('--n-bridge', type=int, default=1)
    ap.add_argument('--seeds', type=int, default=5)
    ap.add_argument('--q', type=float, default=0.1)
    ap.add_argument('--max-steps', type=int, default=2_000_000)
    ap.add_argument('--fresh', action='store_true')
    args = ap.parse_args()

    delta = 1e-3
    seeds = list(range(args.seeds))
    tag = f'fb_smooth_{args.instance}_n{args.n}'
    out_npz = os.path.join(OUT, f'{tag}_results.npz')

    n_p, n_rho, n_c = len(args.ps), len(args.rhos), len(COMBOS)
    # emp does not depend on rho -> stored at rho index 0 and broadcast
    emp = np.full((n_p, n_c, len(seeds)), np.nan)
    emp_cor = np.zeros((n_p, n_c, len(seeds)), dtype=bool)
    reg = np.full((n_p, n_rho, n_c, len(seeds)), np.nan)
    reg_cor = np.zeros((n_p, n_rho, n_c, len(seeds)), dtype=bool)
    emp_cap = np.zeros((n_p, n_c, len(seeds)), dtype=bool)
    reg_cap = np.zeros((n_p, n_rho, n_c, len(seeds)), dtype=bool)
    eps_real = np.full((n_p, len(seeds)), np.nan)
    H_GF = np.full(n_p, np.nan)
    emp_done = np.zeros(n_p, dtype=bool)
    reg_done = np.zeros((n_p, n_rho), dtype=bool)

    def save():
        tmp = out_npz + '.tmp.npz'
        np.savez(tmp, ps=np.array(args.ps), rhos=np.array(args.rhos),
                 seeds=np.array(seeds), combos=np.array([LABEL[c] for c in COMBOS]),
                 emp=emp, emp_correct=emp_cor.astype(int),
                 reg=reg, reg_correct=reg_cor.astype(int),
                 emp_capped=emp_cap.astype(int),
                 reg_capped=reg_cap.astype(int),
                 eps_real=eps_real, H_GF=H_GF,
                 emp_done=emp_done.astype(int), reg_done=reg_done.astype(int),
                 instance=np.array(args.instance), n=int(args.n),
                 gap=float(args.gap), n_bridge=int(args.n_bridge),
                 delta=delta, q=args.q, max_steps=int(args.max_steps))
        os.replace(tmp, out_npz)

    if os.path.exists(out_npz) and not args.fresh:
        try:
            z = np.load(out_npz, allow_pickle=False)
            if (list(z['ps'].tolist()) == list(args.ps)
                    and list(z['rhos'].tolist()) == list(args.rhos)
                    and list(z['seeds'].tolist()) == seeds
                    and str(z['instance']) == args.instance
                    and int(z['n']) == args.n):
                emp, emp_cor = z['emp'], z['emp_correct'].astype(bool)
                reg, reg_cor = z['reg'], z['reg_correct'].astype(bool)
                if 'emp_capped' in z.files:
                    emp_cap = z['emp_capped'].astype(bool)
                    reg_cap = z['reg_capped'].astype(bool)
                eps_real, H_GF = z['eps_real'], z['H_GF']
                emp_done = z['emp_done'].astype(bool)
                reg_done = z['reg_done'].astype(bool)
                print(f"[resume] emp {emp_done.sum()}/{n_p}, "
                      f"reg {reg_done.sum()}/{n_p*n_rho}", flush=True)
        except Exception as e:
            print(f"[resume] failed: {e}", flush=True)

    print(f"[{tag}] instance={args.instance} n={args.n} gap={args.gap} "
          f"n_bridge={args.n_bridge} seeds={len(seeds)}", flush=True)

    def make(kind, stop, pull, rho, mu, A, D):
        def _f():
            return FeedbackGraphBAI(D=D, A=A, mu=mu, delta=delta, q=args.q,
                                    estimator=kind, stop=stop, pull=pull,
                                    rho=(rho if kind == 'reg' else 0.0))
        return _f

    for pi, p in enumerate(args.ps):
        # realized eps / H_GF from seed 0's graph (report per-seed too)
        for si, s in enumerate(seeds):
            mu, A, D = build_instance(args.instance, args.n, p, args.gap,
                                      args.n_bridge, s)
            L = D - A
            eps_real[pi, si] = float(np.sqrt(max(mu @ L @ mu, 0.0)))
        mu0, A0, D0 = build_instance(args.instance, args.n, p, args.gap,
                                     args.n_bridge, 0)
        if np.isnan(H_GF[pi]):
            H_GF[pi] = hardness.graph_feedback_hardness(mu0, A0)
        print(f"\n=== p={p:g}  eps_real={np.median(eps_real[pi]):.3f} "
              f"(range [{eps_real[pi].min():.3f}, {eps_real[pi].max():.3f}])  "
              f"H_GF={H_GF[pi]:.1f} ===", flush=True)

        # --- empirical estimator (rho-independent) ---
        if not emp_done[pi]:
            for ci, (stop, pull) in enumerate(COMBOS):
                ts_, cor_, cap_ = [], [], []
                for s in seeds:
                    mu, A, D = build_instance(args.instance, args.n, p,
                                              args.gap, args.n_bridge, s)
                    r = runners.run_algorithm(
                        make('emp', stop, pull, 0.0, mu, A, D), s,
                        max_steps=args.max_steps, record_elimination=False)
                    ts_.append(r['stopping_time'])
                    cor_.append(r['correct'])
                    cap_.append(not r['converged_flag'])
                emp[pi, ci] = ts_
                emp_cor[pi, ci] = cor_
                emp_cap[pi, ci] = cap_
            emp_done[pi] = True
            save()
        def _cell(v, cor, cap):
            n_cap = int(cap.sum())
            s_ = f"{np.nanmedian(v):,.0f}"
            if n_cap:
                s_ = f"CAP{n_cap}/{len(seeds)}(>={np.nanmedian(v):,.0f})"
            return f"{s_}({cor.mean():.0%})"
        print("  emp: " + "  ".join(
            f"{LABEL[c]}={_cell(emp[pi,ci], emp_cor[pi,ci], emp_cap[pi,ci])}"
            for ci, c in enumerate(COMBOS)), flush=True)

        # --- regularized estimator, per rho ---
        for ri, rho in enumerate(args.rhos):
            if not reg_done[pi, ri]:
                t0 = time.time()
                for ci, (stop, pull) in enumerate(COMBOS):
                    ts_, cor_, cap_ = [], [], []
                    for s in seeds:
                        mu, A, D = build_instance(args.instance, args.n, p,
                                                  args.gap, args.n_bridge, s)
                        r = runners.run_algorithm(
                            make('reg', stop, pull, rho, mu, A, D), s,
                            max_steps=args.max_steps,
                            record_elimination=False)
                        ts_.append(r['stopping_time'])
                        cor_.append(r['correct'])
                        cap_.append(not r['converged_flag'])
                    reg[pi, ri, ci] = ts_
                    reg_cor[pi, ri, ci] = cor_
                    reg_cap[pi, ri, ci] = cap_
                reg_done[pi, ri] = True
                save()
                el = f" ({time.time()-t0:.0f}s)"
            else:
                el = " [resumed]"
            print(f"  reg rho={rho:>6g}: " + "  ".join(
                f"{LABEL[c]}="
                f"{_cell(reg[pi,ri,ci], reg_cor[pi,ri,ci], reg_cap[pi,ri,ci])}"
                for ci, c in enumerate(COMBOS)) + el, flush=True)

    save()

    # ---- the headline contrast --------------------------------------
    print("\n" + "=" * 92)
    print("TS-vs-UCB ratio at matched pull rule  (>1 means UCB wins, "
          "<1 means TS wins)")
    print("=" * 92)
    for pull_name, ci_ts, ci_ucb in [('cover', 0, 1), ('width', 2, 3)]:
        print(f"\n  pull={pull_name}:   TS/UCB stopping-time ratio")
        hdr = f"{'p':>6} {'emp':>9}" + "".join(
            f"{'reg r=' + f'{r:g}':>12}" for r in args.rhos)
        print(hdr)
        for pi, p in enumerate(args.ps):
            e = (np.nanmedian(emp[pi, ci_ts])
                 / max(np.nanmedian(emp[pi, ci_ucb]), 1.0))
            emark = '*' if (emp_cap[pi, ci_ts].any()
                            or emp_cap[pi, ci_ucb].any()) else ' '
            row = f"{p:>6g} {e:>8.3f}{emark}"
            for ri in range(n_rho):
                rr = (np.nanmedian(reg[pi, ri, ci_ts])
                      / max(np.nanmedian(reg[pi, ri, ci_ucb]), 1.0))
                mark = '*' if (reg_cap[pi, ri, ci_ts].any()
                               or reg_cap[pi, ri, ci_ucb].any()) else ' '
                row += f"{rr:>11.3f}{mark}"
            print(row)
    print("\n  * = at least one cell hit the pull cap; the ratio is a bound, "
          "not a measurement.")
    print(f"\nSaved {out_npz}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
