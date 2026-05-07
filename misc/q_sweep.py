from __future__ import annotations

import argparse
import os
import sys
import time
import traceback

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.utils import instances, runners  # noqa: E402
import graph_algo  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs')
os.makedirs(OUT, exist_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--K', type=int, default=50,
                        help="arms in the clustered_chain instance")
    parser.add_argument('--C', type=int, default=2,
                        help="number of clusters in clustered_chain")
    parser.add_argument('--gap-step', type=float, default=0.3)
    parser.add_argument('--rho', type=float, default=100.0)
    parser.add_argument('--qs', type=float, nargs='+',
                        default=[0.01, 0.05, 0.1])
    parser.add_argument('--seeds', type=int, default=5)
    parser.add_argument('--max-steps', type=int, default=2_000_000)
    parser.add_argument('--fresh', action='store_true')
    args = parser.parse_args()

    qs = list(args.qs)
    seeds = list(range(args.seeds))
    delta = 1e-3
    out_npz = os.path.join(OUT, 'q_sweep_results.npz')

    n_q = len(qs)
    stop_times = np.full((n_q, len(seeds)), np.nan)
    correct = np.zeros((n_q, len(seeds)), dtype=bool)
    done = np.zeros((n_q, len(seeds)), dtype=bool)

    def save_checkpoint():
        kwargs = dict(
            qs=np.array(qs),
            seeds=np.array(seeds),
            K=int(args.K),
            C=int(args.C),
            gap_step=float(args.gap_step),
            rho=float(args.rho),
            delta=delta,
            done=done.astype(int),
            stop_times=stop_times,
            correct=correct.astype(int),
        )
        tmp = out_npz + '.tmp.npz'
        np.savez(tmp, **kwargs)
        os.replace(tmp, out_npz)

    if os.path.exists(out_npz) and not args.fresh:
        try:
            z = np.load(out_npz, allow_pickle=False)
            prev_qs = list(z['qs'].tolist())
            prev_seeds = list(z['seeds'].tolist())
            if (prev_qs == qs and prev_seeds == seeds
                    and int(z['K']) == args.K
                    and int(z['C']) == args.C
                    and abs(float(z['gap_step']) - args.gap_step) < 1e-9
                    and abs(float(z['rho']) - args.rho) < 1e-9):
                done = z['done'].astype(bool)
                stop_times = z['stop_times']
                correct = z['correct'].astype(bool)
                print(f"[resume] loaded {int(done.sum())}/"
                      f"{n_q * len(seeds)} cells", flush=True)
            else:
                print("[resume] checkpoint mismatch; starting fresh",
                      flush=True)
        except Exception as e:
            print(f"[resume] failed to load checkpoint: {e}", flush=True)

    print(f"clustered_chain K={args.K}, C={args.C}, "
          f"gap_step={args.gap_step}, rho={args.rho}", flush=True)
    try:
        mu, A, D = instances.clustered_chain(args.K, C=args.C,
                                             gap_step=args.gap_step)
    except Exception as e:
        print(f"failed to build instance: {e}", flush=True)
        traceback.print_exc()
        sys.exit(1)

    for qi, q in enumerate(qs):
        print(f"\n=== q={q} ===", flush=True)
        for si, seed in enumerate(seeds):
            if done[qi, si]:
                continue
            def factory(D=D, A=A, mu=mu, q=q):
                return graph_algo.ThompsonSampling(
                    D=D, A=A, mu=mu, rho_lap=args.rho, delta=delta, q=q)
            t0 = time.time()
            try:
                out = runners.run_algorithm(
                    factory, seed=seed, max_steps=args.max_steps,
                    record_elimination=False)
                stop_times[qi, si] = out['stopping_time']
                correct[qi, si] = out['correct']
                done[qi, si] = True
            except Exception as e:
                print(f"  [skip] q={q} seed={seed} failed: {e}", flush=True)
                traceback.print_exc()
                stop_times[qi, si] = np.nan
                correct[qi, si] = False
                done[qi, si] = True
            save_checkpoint()
            print(f"  q={q} seed={seed} t={stop_times[qi, si]:.0f} "
                  f"correct={int(correct[qi, si])} "
                  f"({time.time() - t0:.1f}s)", flush=True)

    print(f"\nSaved {out_npz}")
    print()
    print("# q sweep summary")
    print(f"{'q':>6}  {'1/log(1/q)':>11}  {'med':>10}  {'p25':>10}  {'p75':>10}")
    for qi, q in enumerate(qs):
        med = np.nanmedian(stop_times[qi])
        p25 = np.nanpercentile(stop_times[qi], 25)
        p75 = np.nanpercentile(stop_times[qi], 75)
        recip = 1.0 / np.log(1.0 / q)
        print(f"{q:>6.3f}  {recip:>11.3f}  {med:>10,.0f}  {p25:>10,.0f}  {p75:>10,.0f}")


if __name__ == "__main__":
    main()
