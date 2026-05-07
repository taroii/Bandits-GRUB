from __future__ import annotations

import argparse
import os
import sys
import time
import traceback

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.utils import instances, hardness, runners  # noqa: E402
import graph_algo  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs')
os.makedirs(OUT, exist_ok=True)


ALGOS = ['TS_tuned', 'TS_rho1', 'Basic']


def build_factory(name, D, A, mu, delta, q, rho_lap=None,
                  epsilon_nominal=None, rho_diag=1e-4):
    if name == 'TS_tuned':
        kw = dict(D=D, A=A, mu=mu, rho_lap=rho_lap, delta=delta, q=q,
                  epsilon_nominal=epsilon_nominal, rho_diag=rho_diag)
        return lambda: graph_algo.ThompsonSampling(**kw)
    if name == 'TS_rho1':
        return lambda: graph_algo.ThompsonSampling(
            D=D, A=A, mu=mu, rho_lap=1.0, delta=delta, q=q)
    if name == 'Basic':
        return lambda: graph_algo.BasicThompsonSampling(
            mu=mu, delta=delta, q=q)
    raise ValueError(name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', type=int, default=5)
    parser.add_argument('--c0', type=float, default=4.0)
    parser.add_argument('--q', type=float, default=0.1)
    parser.add_argument('--max-steps', type=int, default=2_000_000)
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--fresh', action='store_true')
    args = parser.parse_args()

    if args.quick:
        eps_sweep = [1.0, 10 ** -1.0, 10 ** -2.0]
    else:
        eps_sweep = [10 ** (-k / 2.0) for k in range(0, 7)]
    seeds = list(range(args.seeds))
    delta = 1e-3
    out_npz = os.path.join(OUT, 'mis_2_results.npz')

    try:
        mu, A, D = instances.sbm_phase_transition_connected(seed=0)
    except Exception as e:
        print(f"failed to build connected SBM: {e}", flush=True)
        traceback.print_exc()
        sys.exit(1)
    K = len(mu)
    print(f"[mis_2] K={K}, edges={int(A.sum() / 2)} (connected SBM)", flush=True)
    a_star = int(np.argmax(mu))
    Delta_pos = (mu[a_star] - mu)[mu < mu[a_star]]
    print(f"[mis_2] sum 1/D^2={(1.0 / Delta_pos ** 2).sum():.2f}, "
          f"max 1/D^2={(1.0 / Delta_pos ** 2).max():.2f}", flush=True)

    T_est = hardness.classical_hardness(mu) * np.log(1 / delta)
    eps_critical = hardness.critical_epsilons(mu, A, D, T_est, delta,
                                              c0=args.c0)
    H_eps_vals = [hardness.epsilon_hardness(mu, A, D, eps, T_est, delta,
                                            c0=args.c0)
                  for eps in eps_sweep]
    comp_size = []
    for eps in eps_sweep:
        Hs, Ns = hardness.competitive_set_epsilon(mu, A, D, eps, T_est,
                                                  delta, c0=args.c0)
        comp_size.append(len(Hs))

    n_eps = len(eps_sweep)
    stop_times = {a: np.full((n_eps, len(seeds)), np.nan) for a in ALGOS}
    correct = {a: np.zeros((n_eps, len(seeds)), dtype=bool) for a in ALGOS}
    # done is over (eps, algo, seed); the eps-independent baselines
    # (TS_rho1, Basic) are run once and broadcast -- we track them
    # under eps index 0 and copy across at save time.
    done = np.zeros((n_eps, len(ALGOS), len(seeds)), dtype=bool)

    def save_checkpoint():
        # Broadcast eps-independent baselines across eps before saving.
        for ai, name in enumerate(ALGOS):
            if name in ('TS_rho1', 'Basic'):
                # If row 0 has any value, copy to all rows.
                if np.any(done[0, ai, :]):
                    for ei in range(n_eps):
                        for si in range(len(seeds)):
                            if done[0, ai, si] and not done[ei, ai, si]:
                                stop_times[name][ei, si] = stop_times[name][0, si]
                                correct[name][ei, si] = correct[name][0, si]
                                done[ei, ai, si] = done[0, ai, si]
        kwargs = dict(
            eps=np.array(eps_sweep),
            seeds=np.array(seeds),
            H_eps=np.array(H_eps_vals),
            comp_size=np.array(comp_size),
            eps_critical=np.array([eps_critical[i] for i in sorted(eps_critical)]),
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
            prev_eps = list(z['eps'].tolist())
            prev_seeds = list(z['seeds'].tolist())
            if (np.allclose(prev_eps, eps_sweep)
                    and prev_seeds == seeds):
                done = z['done'].astype(bool)
                for a in ALGOS:
                    stop_times[a] = z[f'{a}_stop']
                    correct[a] = z[f'{a}_correct'].astype(bool)
                print(f"[resume] loaded {int(done.sum())}/"
                      f"{n_eps * len(ALGOS) * len(seeds)} cells",
                      flush=True)
            else:
                print("[resume] checkpoint mismatch; starting fresh "
                      "(use --fresh to silence)", flush=True)
        except Exception as e:
            print(f"[resume] failed to load checkpoint: {e}", flush=True)

    # TS_rho1 and Basic: eps-independent, run once at eps index 0 
    for ai, name in enumerate(['TS_rho1', 'Basic']):
        ai_global = ALGOS.index(name)
        for si, seed in enumerate(seeds):
            if done[0, ai_global, si]:
                continue
            fac = build_factory(name, D, A, mu, delta, args.q)
            t0 = time.time()
            try:
                out = runners.run_algorithm(
                    fac, seed=seed, max_steps=args.max_steps,
                    record_elimination=False)
                stop_times[name][0, si] = out['stopping_time']
                correct[name][0, si] = out['correct']
                done[0, ai_global, si] = True
            except Exception as e:
                print(f"  [skip] {name} seed={seed}: {e}", flush=True)
                traceback.print_exc()
                stop_times[name][0, si] = np.nan
                correct[name][0, si] = False
                done[0, ai_global, si] = True
            save_checkpoint()
            print(f"  baseline {name:8s} seed={seed} t={stop_times[name][0, si]:.0f} "
                  f"correct={int(correct[name][0, si])} "
                  f"({time.time() - t0:.1f}s)", flush=True)

    # TS_tuned: eps-dependent main sweep
    ai_tuned = ALGOS.index('TS_tuned')
    for ei, eps in enumerate(eps_sweep):
        rho_star = hardness.rho_star(eps, K, T_est, delta)
        rho_diag_val = max(1e-4, 1e-6 * rho_star)
        print(f"\n=== eps={eps:.3e} rho*={rho_star:.3e} "
              f"H_eps={H_eps_vals[ei]:.2f} |comp|={comp_size[ei]} ===",
              flush=True)
        for si, seed in enumerate(seeds):
            if done[ei, ai_tuned, si]:
                continue
            fac = build_factory('TS_tuned', D, A, mu, delta, args.q,
                                rho_lap=rho_star, epsilon_nominal=eps,
                                rho_diag=rho_diag_val)
            t0 = time.time()
            try:
                out = runners.run_algorithm(
                    fac, seed=seed, max_steps=args.max_steps,
                    record_elimination=False)
                stop_times['TS_tuned'][ei, si] = out['stopping_time']
                correct['TS_tuned'][ei, si] = out['correct']
                done[ei, ai_tuned, si] = True
            except Exception as e:
                print(f"  [skip] TS_tuned eps={eps:.3e} seed={seed}: {e}",
                      flush=True)
                traceback.print_exc()
                stop_times['TS_tuned'][ei, si] = np.nan
                correct['TS_tuned'][ei, si] = False
                done[ei, ai_tuned, si] = True
            save_checkpoint()
            print(f"  TS_tuned eps={eps:.3e} seed={seed} "
                  f"t={stop_times['TS_tuned'][ei, si]:.0f} "
                  f"correct={int(correct['TS_tuned'][ei, si])} "
                  f"({time.time() - t0:.1f}s)", flush=True)

    print(f"\nSaved {out_npz}")
    print("\n# Per-eps medians")
    print(f"{'eps':>10}  " + "  ".join(f"{a:>14}" for a in ALGOS))
    for ei, eps in enumerate(eps_sweep):
        row = f"{eps:>10.3e}  "
        row += "  ".join(
            f"{np.nanmedian(stop_times[a][ei, :]):>14,.0f}" for a in ALGOS)
        print(row)


if __name__ == "__main__":
    main()
