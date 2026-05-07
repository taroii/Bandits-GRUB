from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.utils import hardness, runners, movielens  # noqa: E402
import graph_algo  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs')
os.makedirs(OUT, exist_ok=True)


def build_factory(name, D, A, mu, delta, q, rho, reward_fn=None):
    if name == 'TS-Explore':
        return lambda: graph_algo.ThompsonSampling(
            D=D, A=A, mu=mu, rho_lap=rho, delta=delta, q=q,
            reward_fn=reward_fn)
    if name == 'GRUB':
        return lambda: graph_algo.MaxVarianceArmAlgo(
            D=D, A=A, mu=mu, rho_lap=rho, delta=delta,
            reward_fn=reward_fn)
    if name == 'Basic TS':
        return lambda: graph_algo.BasicThompsonSampling(
            mu=mu, delta=delta, q=q, reward_fn=reward_fn)
    if name == 'KL-LUCB':
        return lambda: graph_algo.KL_LUCB(
            mu=mu, delta=delta, reward_fn=reward_fn)
    raise ValueError(name)


def default_configs():
    """Return list of (label, kwargs-for-build_instance, K) tuples."""
    cfgs = []
    # (a) top-k sweep at K=20
    for tk in [3, 5, 10, 20]:
        cfgs.append((
            f'topk={tk}',
            dict(K=20, top_k_neighbors=tk, min_common=5),
        ))
    # (b) K sweep at top-k=5
    for K in [10, 15]:
        cfgs.append((
            f'K={K}',
            dict(K=K, top_k_neighbors=5, min_common=5),
        ))
    return cfgs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rho', type=float, default=1000.0,
                        help="fixed Laplacian weight; default rho=1000 is "
                             "the regime where movielens_1 found the "
                             "biggest gap to Basic TS")
    parser.add_argument('--seeds', type=int, default=10)
    parser.add_argument('--q', type=float, default=0.1)
    parser.add_argument('--max-steps', type=int, default=10_000_000)
    parser.add_argument('--reward-model', type=str, default='empirical',
                        choices=['empirical', 'gaussian'])
    parser.add_argument('--include-grub', action='store_true',
                        help="also run GRUB (slow)")
    parser.add_argument('--include-kllucb', action='store_true',
                        help="also run KL-LUCB")
    parser.add_argument('--quick', action='store_true',
                        help="seeds=3, configs=[topk=5/K=20, topk=10/K=20], "
                             "max-steps=2_000_000")
    parser.add_argument('--fresh', action='store_true')
    args = parser.parse_args()

    configs = default_configs()
    if args.quick:
        configs = [
            ('topk=5',  dict(K=20, top_k_neighbors=5, min_common=5)),
            ('topk=10', dict(K=20, top_k_neighbors=10, min_common=5)),
        ]
        args.seeds = min(args.seeds, 3) if args.seeds > 3 else args.seeds
        args.max_steps = min(args.max_steps, 2_000_000)

    seeds = list(range(args.seeds))
    delta = 1e-3
    out_npz = os.path.join(OUT, 'movielens_ablations_results.npz')

    algos = ['TS-Explore', 'Basic TS']
    if args.include_grub:
        algos.append('GRUB')
    if args.include_kllucb:
        algos.append('KL-LUCB')

    n_cfg = len(configs)
    stop_times = {a: np.full((n_cfg, len(seeds)), np.nan) for a in algos}
    correct = {a: np.zeros((n_cfg, len(seeds)), dtype=bool) for a in algos}
    cfg_eps = np.full(n_cfg, np.nan)
    cfg_H_cls = np.full(n_cfg, np.nan)
    cfg_gap_min = np.full(n_cfg, np.nan)
    cfg_n_edges = np.full(n_cfg, np.nan)
    done = np.zeros((n_cfg, len(algos)), dtype=bool)

    def save_checkpoint():
        kwargs = dict(
            labels=np.array([c[0] for c in configs]),
            algo_names=np.array(algos),
            seeds=np.array(seeds),
            rho=float(args.rho),
            reward_model=np.array(args.reward_model),
            delta=delta,
            q=args.q,
            cfg_eps=cfg_eps,
            cfg_H_cls=cfg_H_cls,
            cfg_gap_min=cfg_gap_min,
            cfg_n_edges=cfg_n_edges,
            done=done.astype(int),
        )
        for a in algos:
            kwargs[f'{a}_stop'] = stop_times[a]
            kwargs[f'{a}_correct'] = correct[a].astype(int)
        tmp = out_npz + '.tmp.npz'
        np.savez(tmp, **kwargs)
        os.replace(tmp, out_npz)

    if os.path.exists(out_npz) and not args.fresh:
        try:
            z = np.load(out_npz, allow_pickle=False)
            prev_labels = list(z['labels'].tolist())
            prev_algos = list(z['algo_names'].tolist())
            prev_seeds = list(z['seeds'].tolist())
            prev_rho = float(z['rho'])
            prev_reward = (str(z['reward_model'])
                           if 'reward_model' in z.files else 'gaussian')
            cur_labels = [c[0] for c in configs]
            if (prev_labels == cur_labels and prev_algos == algos
                    and prev_seeds == seeds
                    and abs(prev_rho - args.rho) < 1e-9
                    and prev_reward == args.reward_model):
                done = z['done'].astype(bool)
                cfg_eps = z['cfg_eps']
                cfg_H_cls = z['cfg_H_cls']
                cfg_gap_min = z['cfg_gap_min']
                cfg_n_edges = z['cfg_n_edges']
                for a in algos:
                    stop_times[a] = z[f'{a}_stop']
                    correct[a] = z[f'{a}_correct'].astype(bool)
                print(f"[resume] loaded {int(done.sum())}/"
                      f"{n_cfg*len(algos)} cells", flush=True)
            else:
                print(f"[resume] checkpoint mismatch; ignoring "
                      f"(use --fresh to silence)", flush=True)
        except Exception as e:
            print(f"[resume] failed to load {out_npz}: {e}", flush=True)

    # Main loop: each config builds its own instance + reward sampler.
    for ci, (label, build_kwargs) in enumerate(configs):
        print(f"\n=== config[{ci}] {label}  "
              f"({build_kwargs}) ===", flush=True)
        mu, A, D, meta = movielens.build_instance(
            return_meta=True, **build_kwargs)
        L = D - A
        eps = float(np.sqrt(max(mu @ L @ mu, 0.0)))
        H_cls = hardness.classical_hardness(mu)
        n_edges = int(A.sum() / 2)
        a_star = int(np.argmax(mu))
        Delta_pos = (mu[a_star] - mu)[mu < mu[a_star]]
        gap_min = float(Delta_pos.min())
        cfg_eps[ci] = eps
        cfg_H_cls[ci] = H_cls
        cfg_gap_min[ci] = gap_min
        cfg_n_edges[ci] = n_edges
        print(f"  K={len(mu)}, edges={n_edges}, eps={eps:.3f}, "
              f"H_cls={H_cls:.1f}, gap_min={gap_min:.4f}", flush=True)

        if args.reward_model == 'empirical':
            reward_fn = movielens.make_empirical_reward_fn(
                meta['ratings_per_arm'])
        else:
            reward_fn = None

        for ai, name in enumerate(algos):
            if done[ci, ai]:
                ts = stop_times[name][ci, :]
                print(f"  {name:11s} [resumed] t_med={np.nanmedian(ts):.0f}",
                      flush=True)
                continue
            fac = build_factory(name, D, A, mu, delta, args.q,
                                rho=args.rho, reward_fn=reward_fn)
            t0 = time.time()
            runs = runners.run_many(fac, seeds,
                                    max_steps=args.max_steps,
                                    record_elimination=False, progress=False)
            ts = np.array([r['stopping_time'] for r in runs], dtype=float)
            cor = np.array([r['correct'] for r in runs], dtype=bool)
            stop_times[name][ci, :] = ts
            correct[name][ci, :] = cor
            done[ci, ai] = True
            save_checkpoint()
            print(f"  {name:11s} t_med={np.median(ts):8.0f}  "
                  f"IQR=[{np.percentile(ts,25):.0f}, "
                  f"{np.percentile(ts,75):.0f}]  "
                  f"correct={cor.mean():.0%}  ({time.time()-t0:.0f}s)",
                  flush=True)

    print(f"\nSaved {out_npz}")
    print()
    print("# Ablation summary")
    header = f"{'config':<12s}  " + '  '.join(
        f"{a+' med':>15s}" for a in algos)
    print(header)
    print('-' * len(header))
    for ci, (label, _) in enumerate(configs):
        row = f"{label:<12s}  "
        row += '  '.join(
            f"{np.nanmedian(stop_times[a][ci, :]):>15,.0f}" for a in algos)
        print(row)


if __name__ == "__main__":
    main()
