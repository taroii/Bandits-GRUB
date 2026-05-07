from __future__ import annotations

import argparse
import os
import sys
import time
import traceback

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.utils import hardness, runners, movielens  # noqa: E402
import graph_algo  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs')
os.makedirs(OUT, exist_ok=True)


ALGOS = ['TS-Explore', 'Basic TS']


def build_factory(name, D, A, mu, delta, q, rho, reward_fn=None):
    if name == 'TS-Explore':
        return lambda: graph_algo.ThompsonSampling(
            D=D, A=A, mu=mu, rho_lap=rho, delta=delta, q=q,
            reward_fn=reward_fn)
    if name == 'Basic TS':
        return lambda: graph_algo.BasicThompsonSampling(
            mu=mu, delta=delta, q=q, reward_fn=reward_fn)
    raise ValueError(name)


def default_configs():
    """Three top-k values at K=20, the headline configuration."""
    return [
        ('topk=3',  dict(K=20, top_k_neighbors=3,  min_common=5)),
        ('topk=5',  dict(K=20, top_k_neighbors=5,  min_common=5)),
        ('topk=10', dict(K=20, top_k_neighbors=10, min_common=5)),
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rho', type=float, default=1000.0,
                        help="fixed Laplacian weight (default 10^3, "
                             "matching Figure 1 right empirical optimum)")
    parser.add_argument('--seeds', type=int, default=5)
    parser.add_argument('--q', type=float, default=0.1)
    parser.add_argument('--max-steps', type=int, default=1_200_000,
                        help="per-cell pull cap (default 1.2M). Bounds each "
                             "Basic TS cell to ~3-4 min of compute on this "
                             "instance to limit crash exposure on machines "
                             "with the deep-C-state reboot bug; some seeds "
                             "will report 'didn't converge', which only "
                             "strengthens the headline TS-Explore vs Basic "
                             "TS comparison.")
    parser.add_argument('--reward-model', type=str, default='empirical',
                        choices=['empirical', 'gaussian'])
    parser.add_argument('--fresh', action='store_true')
    args = parser.parse_args()

    configs = default_configs()
    seeds = list(range(args.seeds))
    delta = 1e-3
    out_npz = os.path.join(OUT, 'movielens_robustness_results.npz')

    n_cfg = len(configs)
    stop_times = {a: np.full((n_cfg, len(seeds)), np.nan) for a in ALGOS}
    correct = {a: np.zeros((n_cfg, len(seeds)), dtype=bool) for a in ALGOS}
    cfg_eps = np.full(n_cfg, np.nan)
    cfg_H_cls = np.full(n_cfg, np.nan)
    cfg_gap_min = np.full(n_cfg, np.nan)
    cfg_n_edges = np.full(n_cfg, np.nan)
    # Per-(config, algo, seed) done flag so we can resume mid-cell.
    done = np.zeros((n_cfg, len(ALGOS), len(seeds)), dtype=bool)

    def save_checkpoint():
        kwargs = dict(
            labels=np.array([c[0] for c in configs]),
            algo_names=np.array(ALGOS),
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
        for a in ALGOS:
            kwargs[f'{a}_stop'] = stop_times[a]
            kwargs[f'{a}_correct'] = correct[a].astype(int)
        # Atomic write with retry: on Windows, OneDrive / antivirus / an
        # open Explorer preview can briefly hold the destination file and
        # cause `os.replace` to raise PermissionError [WinError 5]. Retry
        # with linear backoff (~18s total) so transient locks don't kill
        # an hour of compute.
        tmp = out_npz + '.tmp.npz'
        last_err = None
        for attempt in range(8):
            try:
                np.savez(tmp, **kwargs)
                os.replace(tmp, out_npz)
                return
            except PermissionError as e:
                last_err = e
                time.sleep(0.5 + 0.5 * attempt)
        print(f"  [warn] checkpoint save failed after 8 retries: "
              f"{last_err}", flush=True)

    if os.path.exists(out_npz) and not args.fresh:
        try:
            z = np.load(out_npz, allow_pickle=False)
            prev_labels = list(z['labels'].astype(str).tolist())
            prev_algos = list(z['algo_names'].astype(str).tolist())
            prev_seeds = list(z['seeds'].tolist())
            prev_rho = float(z['rho'])
            prev_reward = (str(z['reward_model'])
                           if 'reward_model' in z.files else 'gaussian')
            cur_labels = [c[0] for c in configs]
            if (prev_labels == cur_labels
                    and prev_algos == ALGOS
                    and prev_seeds == seeds
                    and abs(prev_rho - args.rho) < 1e-9
                    and prev_reward == args.reward_model):
                done = z['done'].astype(bool)
                cfg_eps = z['cfg_eps']
                cfg_H_cls = z['cfg_H_cls']
                cfg_gap_min = z['cfg_gap_min']
                cfg_n_edges = z['cfg_n_edges']
                for a in ALGOS:
                    stop_times[a] = z[f'{a}_stop']
                    correct[a] = z[f'{a}_correct'].astype(bool)
                print(f"[resume] loaded {int(done.sum())}/"
                      f"{n_cfg * len(ALGOS) * len(seeds)} cells",
                      flush=True)
            else:
                print("[resume] checkpoint mismatch; starting fresh "
                      "(use --fresh to silence)", flush=True)
        except Exception as e:
            print(f"[resume] failed to load checkpoint: {e}", flush=True)

    for ci, (label, build_kwargs) in enumerate(configs):
        print(f"\n=== config[{ci}] {label}  "
              f"({build_kwargs}) ===", flush=True)
        try:
            mu, A, D, meta = movielens.build_instance(
                return_meta=True, **build_kwargs)
        except Exception as e:
            print(f"  [skip] failed to build instance: {e}", flush=True)
            traceback.print_exc()
            continue

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

        for ai, name in enumerate(ALGOS):
            for si, seed in enumerate(seeds):
                if done[ci, ai, si]:
                    continue
                fac = build_factory(name, D, A, mu, delta, args.q,
                                    rho=args.rho, reward_fn=reward_fn)
                t0 = time.time()
                try:
                    out = runners.run_algorithm(
                        fac, seed=seed, max_steps=args.max_steps,
                        record_elimination=False)
                    stop_times[name][ci, si] = out['stopping_time']
                    correct[name][ci, si] = out['correct']
                    done[ci, ai, si] = True
                except Exception as e:
                    print(f"  [skip seed] {name} seed={seed} failed: {e}",
                          flush=True)
                    traceback.print_exc()
                    # Mark done with NaN to avoid retrying a deterministic
                    # failure on resume.
                    stop_times[name][ci, si] = np.nan
                    correct[name][ci, si] = False
                    done[ci, ai, si] = True
                save_checkpoint()
                print(f"  {name:11s} seed={seed} t={stop_times[name][ci, si]:.0f} "
                      f"correct={int(correct[name][ci, si])} "
                      f"({time.time() - t0:.1f}s)", flush=True)

    print(f"\nSaved {out_npz}")
    print()
    print("# Robustness summary (median stopping time per algorithm, per config)")
    header = f"{'config':<12s}  " + '  '.join(
        f"{a + ' med':>16s}" for a in ALGOS)
    print(header)
    print('-' * len(header))
    for ci, (label, _) in enumerate(configs):
        row = f"{label:<12s}  "
        row += '  '.join(
            f"{np.nanmedian(stop_times[a][ci, :]):>16,.0f}" for a in ALGOS)
        print(row)


if __name__ == "__main__":
    main()
