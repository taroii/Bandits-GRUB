"""Task 0 -- rerun the GRUB cells of Figure 1 under each bias convention.

Writes one .npz per (instance, bias) so the committed
``main_2_results.npz`` / ``movielens_1_results.npz`` are never touched.

  python experiments/grub_bias_sweep.py --instance chain --bias published
  python experiments/grub_bias_sweep.py --instance movielens --bias sqrt

Checkpointed per (cell) so a crash or Ctrl-C resumes from the last
completed cell.  Cap-hits are recorded explicitly rather than being
reported as stopping times.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.utils import instances, hardness, runners  # noqa: E402
import graph_algo  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs')
os.makedirs(OUT, exist_ok=True)

BIASES = ('legacy', 'published', 'sqrt', 'oracle', 'none')


def grub_factory(D, A, mu, delta, rho, bias, log_base, reward_fn=None):
    def _make():
        return graph_algo.MaxVarianceArmAlgo(
            D=D, A=A, mu=mu, rho_lap=rho, delta=delta,
            grub_bias=bias, grub_log=log_base, reward_fn=reward_fn)
    return _make


def run_cell(fac, seeds, max_steps):
    runs = runners.run_many(fac, seeds, max_steps=max_steps,
                            record_elimination=False, progress=False)
    ts = np.array([r['stopping_time'] for r in runs], dtype=float)
    cor = np.array([r['correct'] for r in runs], dtype=bool)
    conv = np.array([r['converged_flag'] for r in runs], dtype=bool)
    return ts, cor, conv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--instance', choices=['chain', 'movielens'],
                    default='chain')
    ap.add_argument('--bias', choices=BIASES, default='published')
    ap.add_argument('--log', choices=['auto', 'ln', 'log2'], default='auto')
    ap.add_argument('--seeds', type=int, default=5,
                    help="pilot at 5 (plan default); top up later")
    ap.add_argument('--max-steps', type=int, default=10_000_000)
    ap.add_argument('--Ks', type=int, nargs='+',
                    default=[10, 20, 50, 100, 200],
                    help="chain instance only")
    ap.add_argument('--rhos', type=float, nargs='+',
                    default=[1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0],
                    help="movielens instance only")
    ap.add_argument('--rho', type=float, default=100.0,
                    help="chain instance only")
    ap.add_argument('--K', type=int, default=20,
                    help="movielens instance only")
    ap.add_argument('--gap-step', type=float, default=0.3)
    ap.add_argument('--C', type=int, default=2)
    ap.add_argument('--fresh', action='store_true')
    args = ap.parse_args()

    delta = 1e-3
    seeds = list(range(args.seeds))
    tag = f"grub_bias_{args.instance}_{args.bias}"
    if args.log != 'auto':
        tag += f"_log{args.log}"
    out_npz = os.path.join(OUT, f'{tag}_results.npz')

    # --- build the cell axis -------------------------------------------
    if args.instance == 'chain':
        axis = list(args.Ks)
        axis_name = 'K'
    else:
        axis = list(args.rhos)
        axis_name = 'rho'

    n_cells = len(axis)
    stop = np.full((n_cells, len(seeds)), np.nan)
    correct = np.zeros((n_cells, len(seeds)), dtype=bool)
    capped = np.zeros((n_cells, len(seeds)), dtype=bool)
    eps_vals = np.full(n_cells, np.nan)
    done = np.zeros(n_cells, dtype=bool)

    def save():
        tmp = out_npz + '.tmp.npz'
        np.savez(tmp, axis=np.array(axis, dtype=float),
                 axis_name=np.array(axis_name), seeds=np.array(seeds),
                 stop=stop, correct=correct.astype(int),
                 capped=capped.astype(int), eps=eps_vals,
                 done=done.astype(int), bias=np.array(args.bias),
                 log_base=np.array(args.log), delta=delta,
                 max_steps=int(args.max_steps),
                 instance=np.array(args.instance),
                 rho=float(args.rho), K=int(args.K),
                 gap_step=float(args.gap_step), C=int(args.C))
        os.replace(tmp, out_npz)

    if os.path.exists(out_npz) and not args.fresh:
        try:
            z = np.load(out_npz, allow_pickle=False)
            if (list(z['axis'].tolist()) == [float(a) for a in axis]
                    and list(z['seeds'].tolist()) == seeds
                    and str(z['bias']) == args.bias
                    and int(z['max_steps']) == int(args.max_steps)):
                stop = z['stop']
                correct = z['correct'].astype(bool)
                capped = z['capped'].astype(bool)
                eps_vals = z['eps']
                done = z['done'].astype(bool)
                print(f"[resume] {int(done.sum())}/{n_cells} cells done",
                      flush=True)
            else:
                print("[resume] checkpoint mismatch; ignoring", flush=True)
        except Exception as e:
            print(f"[resume] load failed: {e}", flush=True)

    print(f"[{tag}] instance={args.instance} bias={args.bias} "
          f"log={args.log} seeds={len(seeds)} max_steps={args.max_steps:,}",
          flush=True)

    # --- MovieLens instance is built once -------------------------------
    reward_fn = None
    if args.instance == 'movielens':
        from experiments.utils import movielens
        mu, A, D, meta = movielens.build_instance(
            K=args.K, top_k_neighbors=5, min_common=5, return_meta=True)
        reward_fn = movielens.make_empirical_reward_fn(meta['ratings_per_arm'])
        L = D - A
        print(f"  movielens K={args.K}  eps={np.sqrt(max(mu@L@mu,0)):.3f}  "
              f"gap_min={np.min((mu.max()-mu)[mu<mu.max()]):.4f}", flush=True)

    for ci, cell in enumerate(axis):
        if args.instance == 'chain':
            K = int(cell)
            rho = args.rho
            mu, A, D = instances.clustered_chain(K, C=args.C,
                                                 gap_step=args.gap_step)
        else:
            rho = float(cell)

        L = D - A
        if np.isnan(eps_vals[ci]):
            eps_vals[ci] = float(np.sqrt(max(mu @ L @ mu, 0.0)))

        label = f"{axis_name}={cell:g}"
        if done[ci]:
            ts = stop[ci]
            print(f"  {label:>12s} [resumed] t_med={np.nanmedian(ts):10,.0f}  "
                  f"capped={capped[ci].sum()}/{len(seeds)}", flush=True)
            continue

        fac = grub_factory(D, A, mu, delta, rho, args.bias, args.log,
                           reward_fn=reward_fn)
        t0 = time.time()
        ts, cor, conv = run_cell(fac, seeds, args.max_steps)
        stop[ci] = ts
        correct[ci] = cor
        capped[ci] = ~conv
        done[ci] = True
        save()
        n_cap = int((~conv).sum())
        print(f"  {label:>12s} t_med={np.median(ts):10,.0f}  "
              f"IQR=[{np.percentile(ts,25):,.0f}, {np.percentile(ts,75):,.0f}]  "
              f"correct={cor.mean():.0%}  capped={n_cap}/{len(seeds)}  "
              f"eps={eps_vals[ci]:.3f}  ({time.time()-t0:.0f}s)", flush=True)

    save()
    print(f"\nSaved {out_npz}", flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
