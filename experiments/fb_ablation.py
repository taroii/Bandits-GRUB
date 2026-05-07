from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.utils import instances, runners  # noqa: E402
import graph_algo  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs')
os.makedirs(OUT, exist_ok=True)


# Algorithm names used as keys in the saved npz.  Order matters: it sets
# the legend order in the plot.
ALGOS = ['TS+cover', 'TS+width', 'UCB+cover', 'UCB+width']


def build_factory(name, D, A, mu, delta, q):
    if name == 'TS+cover':
        return lambda: graph_algo.GraphFeedbackTS(
            D=D, A=A, mu=mu, delta=delta, q=q)
    if name == 'TS+width':
        return lambda: graph_algo.GraphFeedbackTSWidth(
            D=D, A=A, mu=mu, delta=delta, q=q)
    if name == 'UCB+cover':
        return lambda: graph_algo.UCBNCover(
            D=D, A=A, mu=mu, delta=delta)
    if name == 'UCB+width':
        return lambda: graph_algo.UCB_N(
            D=D, A=A, mu=mu, delta=delta)
    raise ValueError(name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', type=int, default=20)
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--q', type=float, default=0.1,
                        help="TS tail quantile; q=0.1 is the loosest stopping rule")
    parser.add_argument('--n', type=int, default=20,
                        help="ER graph size (must match fb_1.py for parity)")
    parser.add_argument('--gap', type=float, default=0.3,
                        help="suboptimality gap for every non-optimal arm")
    parser.add_argument('--max-steps', type=int, default=300_000)
    parser.add_argument('--fresh', action='store_true',
                        help="ignore any existing checkpoint and start over")
    args = parser.parse_args()

    if args.quick:
        ps = [0.1, 0.5, 1.0]
        seeds = list(range(max(args.seeds, 3)))
    else:
        ps = [0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
        seeds = list(range(args.seeds))

    delta = 1e-3
    n = args.n
    gap = args.gap
    out_npz = os.path.join(OUT, 'fb_ablation_results.npz')

    # State (with checkpoint resume by (p_index, algo_index) cell).
    stop_times = {a: np.full((len(ps), len(seeds)), np.nan) for a in ALGOS}
    correct = {a: np.zeros((len(ps), len(seeds)), dtype=bool) for a in ALGOS}
    converged = {a: np.zeros((len(ps), len(seeds)), dtype=bool) for a in ALGOS}
    done = np.zeros((len(ps), len(ALGOS)), dtype=bool)

    def save_checkpoint():
        kwargs = dict(
            ps=np.array(ps),
            seeds=np.array(seeds),
            n=n, gap=gap, delta=delta, q=args.q,
            done=done.astype(int),
        )
        for a in ALGOS:
            kwargs[f'{a}_stop'] = stop_times[a]
            kwargs[f'{a}_correct'] = correct[a].astype(int)
            kwargs[f'{a}_converged'] = converged[a].astype(int)
        tmp = out_npz + '.tmp.npz'
        np.savez(tmp, **kwargs)
        os.replace(tmp, out_npz)

    if os.path.exists(out_npz) and not args.fresh:
        try:
            z = np.load(out_npz, allow_pickle=False)
            prev_ps = list(z['ps'].tolist())
            prev_seeds = list(z['seeds'].tolist())
            if (prev_ps == ps and prev_seeds == seeds
                    and int(z['n']) == n
                    and abs(float(z['gap']) - gap) < 1e-9):
                done = z['done'].astype(bool)
                for a in ALGOS:
                    if f'{a}_stop' in z.files:
                        stop_times[a] = z[f'{a}_stop']
                        correct[a] = z[f'{a}_correct'].astype(bool)
                        if f'{a}_converged' in z.files:
                            converged[a] = z[f'{a}_converged'].astype(bool)
                n_done = int(done.sum())
                total = len(ps) * len(ALGOS)
                print(f"[resume] loaded {n_done}/{total} cells", flush=True)
            else:
                print(f"[resume] checkpoint mismatch; ignoring "
                      f"(use --fresh to silence)", flush=True)
        except Exception as e:
            print(f"[resume] failed to load {out_npz}: {e}", flush=True)

    # (p, algo) sweep.  Inner loop = seeds (fresh ER graph per seed,
    # matching fb_1.py).  Each algo gets its own (p, seed) cell so we
    # can checkpoint mid-sweep.
    for pi, p in enumerate(ps):
        print(f"\n=== p={p} ===", flush=True)
        for ai, name in enumerate(ALGOS):
            if done[pi, ai]:
                ts = stop_times[name][pi, :]
                print(f"  {name:11s} [resumed] t_med={np.nanmedian(ts):.0f}",
                      flush=True)
                continue
            t0 = time.time()
            for si, k in enumerate(seeds):
                mu, A, D = instances.erdos_renyi(n=n, p=p, gap=gap, seed=k)
                fac = build_factory(name, D, A, mu, delta, args.q)
                out = runners.run_algorithm(
                    fac, seed=k, max_steps=args.max_steps,
                    record_elimination=False)
                stop_times[name][pi, si] = out['stopping_time']
                correct[name][pi, si] = out['correct']
                converged[name][pi, si] = out['converged_flag']
            done[pi, ai] = True
            save_checkpoint()
            ts = stop_times[name][pi, :]
            cor = correct[name][pi, :]
            conv = converged[name][pi, :]
            n_unconv = int((~conv).sum())
            unconv_str = (f", unconverged={n_unconv}/{len(seeds)}"
                          if n_unconv else "")
            elapsed = time.time() - t0
            print(f"  {name:11s} t_med={np.nanmedian(ts):8.0f}  "
                  f"IQR=[{np.percentile(ts,25):.0f}, "
                  f"{np.percentile(ts,75):.0f}]  "
                  f"correct={cor.mean()*100:.0f}%{unconv_str}  "
                  f"({elapsed:.0f}s)", flush=True)

    print(f"\nSaved {out_npz}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
