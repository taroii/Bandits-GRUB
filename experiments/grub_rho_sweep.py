"""grub_rho_sweep -- 3x3 GRUB sweep over (rho, K) on the synthetic chain.

Companion experiment for Section 4.1 / Appendix experiments. The
existing fig:graph-smooth left panel runs GRUB at fixed rho=100 and
shows linear-in-K scaling consistent with Theorem 4.4 of
\\cite{thaker2022maximizing}. The defense rests on the observation
that at rho=100 the bias term rho*epsilon dominates the GRUB radius
and pushes all K-1 challengers into H_D. To strengthen the defense,
this script sweeps rho in {1, 10, 100} so that we also have GRUB
data in the regime where the variance term dominates: at rho=1, the
bias rho*epsilon is small relative to 2*sigma*sqrt(14 log(.)), and
the parameter-regime prediction is that |H_D| should collapse to a
constant and GRUB should exhibit cluster-aggregate (not linear-in-K)
scaling. The K-sweep at each rho lets us read off the slope.

Instance: clustered_chain (C=2) with singleton best arm at mu=1 and
challenger clique of K-1 vertices at mu=1-Delta with Delta=0.3.
Rewards Gaussian(mu_i, sigma=1). delta=1e-3. Per-run cap 10^7 pulls.
Smoothness epsilon is computed from the true means at instance
construction time (epsilon_nominal=None in MaxVarianceArmAlgo, which
sets self.eps = sqrt(<mu, L_G mu>) = Delta on this chain), so GRUB
is given the correct smoothness and not a misspecified one.

Crash-proof:
  * strictly sequential (no multiprocessing pools),
  * per-cell try/except (one failed seed does not propagate),
  * atomic checkpoint after every (rho, K, seed) cell,
  * resume on restart.

Saves results to experiments/outputs/grub_rho_sweep_results.npz.
"""
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--Ks', type=int, nargs='+',
                        default=[20, 50, 100],
                        help="K values to sweep (default: 20 50 100)")
    parser.add_argument('--rhos', type=float, nargs='+',
                        default=[1.0, 10.0, 100.0],
                        help="rho values to sweep (default: 1 10 100)")
    parser.add_argument('--seeds', type=int, default=20,
                        help="number of seeds per cell (default 20)")
    parser.add_argument('--C', type=int, default=2,
                        help="number of clusters in clustered_chain "
                             "(default 2: best singleton + one challenger "
                             "clique)")
    parser.add_argument('--gap-step', type=float, default=0.3,
                        help="per-cluster gap increment Delta (default 0.3)")
    parser.add_argument('--max-steps', type=int, default=10_000_000,
                        help="hard cap on steps per run (default 10^7)")
    parser.add_argument('--fresh', action='store_true',
                        help="ignore any existing checkpoint and start over")
    args = parser.parse_args()

    Ks = list(args.Ks)
    rhos = list(args.rhos)
    seeds = list(range(args.seeds))
    delta = 1e-3

    n_rho = len(rhos)
    n_K = len(Ks)
    out_npz = os.path.join(OUT, 'grub_rho_sweep_results.npz')

    stop_times = np.full((n_rho, n_K, len(seeds)), np.nan)
    correct = np.zeros((n_rho, n_K, len(seeds)), dtype=bool)
    converged = np.zeros((n_rho, n_K, len(seeds)), dtype=bool)
    H_classical = np.full(n_K, np.nan)
    H_graph = np.full((n_rho, n_K), np.nan)
    eps_realized = np.full(n_K, np.nan)
    done = np.zeros((n_rho, n_K, len(seeds)), dtype=bool)

    def save_checkpoint():
        kwargs = dict(
            rhos=np.array(rhos),
            Ks=np.array(Ks),
            seeds=np.array(seeds),
            delta=float(delta),
            C=int(args.C),
            gap_step=float(args.gap_step),
            max_steps=int(args.max_steps),
            stop_times=stop_times,
            correct=correct.astype(int),
            converged=converged.astype(int),
            H_classical=H_classical,
            H_graph=H_graph,
            eps=eps_realized,
            done=done.astype(int),
        )
        # Atomic write with a small retry loop (defensive against
        # transient filesystem locks on shared servers).
        tmp = out_npz + '.tmp.npz'
        last_err = None
        for attempt in range(5):
            try:
                np.savez(tmp, **kwargs)
                os.replace(tmp, out_npz)
                return
            except (PermissionError, OSError) as e:
                last_err = e
                time.sleep(0.5 + 0.5 * attempt)
        print(f"  [warn] checkpoint save failed after retries: {last_err}",
              flush=True)

    if os.path.exists(out_npz) and not args.fresh:
        try:
            z = np.load(out_npz, allow_pickle=False)
            prev_rhos = list(z['rhos'].tolist())
            prev_Ks = list(z['Ks'].tolist())
            prev_seeds = list(z['seeds'].tolist())
            if (prev_rhos == rhos
                    and prev_Ks == Ks
                    and prev_seeds == seeds
                    and int(z['C']) == int(args.C)
                    and abs(float(z['gap_step']) - float(args.gap_step)) < 1e-9):
                done = z['done'].astype(bool)
                stop_times = z['stop_times']
                correct = z['correct'].astype(bool)
                converged = z['converged'].astype(bool)
                H_classical = z['H_classical']
                H_graph = z['H_graph']
                eps_realized = z['eps']
                print(f"[resume] loaded {int(done.sum())}/"
                      f"{n_rho * n_K * len(seeds)} cells", flush=True)
            else:
                print("[resume] checkpoint mismatch; starting fresh "
                      "(use --fresh to silence)", flush=True)
        except Exception as e:
            print(f"[resume] failed to load checkpoint: {e}", flush=True)

    # ----------------------------------------------------------------
    # Pre-compute instance characteristics (cheap; do once per K).
    # ----------------------------------------------------------------
    print(f"# clustered_chain (C={args.C}, gap_step={args.gap_step}); "
          f"GRUB only; max_steps={args.max_steps:,}", flush=True)
    print(f"{'K':>5}  {'edges':>6}  {'eps':>7}  {'H_classical':>12}",
          flush=True)
    instances_per_K = {}
    for ki, K in enumerate(Ks):
        try:
            mu, A, D = instances.clustered_chain(K, C=args.C,
                                                 gap_step=args.gap_step)
        except Exception as e:
            print(f"  [skip K={K}] failed to build instance: {e}", flush=True)
            traceback.print_exc()
            continue
        instances_per_K[K] = (mu, A, D)
        L = D - A
        eps = float(np.sqrt(max(mu @ L @ mu, 0.0)))
        eps_realized[ki] = eps
        H_classical[ki] = hardness.classical_hardness(mu)
        n_edges = int(A.sum() / 2)
        print(f"{K:>5}  {n_edges:>6}  {eps:>7.3f}  {H_classical[ki]:>12.2f}",
              flush=True)
        for ri, rho in enumerate(rhos):
            H_graph[ri, ki] = hardness.graph_hardness(mu, A, D, rho=rho)

    # ----------------------------------------------------------------
    # Main sweep: (rho, K, seed) cells.
    # ----------------------------------------------------------------
    for ri, rho in enumerate(rhos):
        for ki, K in enumerate(Ks):
            if K not in instances_per_K:
                continue
            mu, A, D = instances_per_K[K]
            print(f"\n=== rho={rho}  K={K}  H_graph={H_graph[ri, ki]:.2f} ===",
                  flush=True)
            for si, seed in enumerate(seeds):
                if done[ri, ki, si]:
                    continue
                # GRUB factory: epsilon_nominal=None lets the algorithm
                # compute the realized smoothness from the true means
                # (which are passed in for reward generation), so GRUB is
                # given the correct smoothness on this instance.
                def factory(D=D, A=A, mu=mu, rho=rho):
                    return graph_algo.MaxVarianceArmAlgo(
                        D=D, A=A, mu=mu, rho_lap=rho, delta=delta)
                t0 = time.time()
                try:
                    out = runners.run_algorithm(
                        factory, seed=seed, max_steps=args.max_steps,
                        record_elimination=False)
                    stop_times[ri, ki, si] = out['stopping_time']
                    correct[ri, ki, si] = out['correct']
                    converged[ri, ki, si] = out['converged_flag']
                    done[ri, ki, si] = True
                except Exception as e:
                    print(f"  [skip] rho={rho} K={K} seed={seed} "
                          f"failed: {e}", flush=True)
                    traceback.print_exc()
                    stop_times[ri, ki, si] = np.nan
                    correct[ri, ki, si] = False
                    converged[ri, ki, si] = False
                    done[ri, ki, si] = True
                save_checkpoint()
                t_str = (f"{stop_times[ri, ki, si]:.0f}"
                         if not np.isnan(stop_times[ri, ki, si]) else "nan")
                conv = "Y" if converged[ri, ki, si] else "N"
                cor = "Y" if correct[ri, ki, si] else "N"
                print(f"  rho={rho:6.1f} K={K:>4d} seed={seed} "
                      f"t={t_str:>10s}  conv={conv}  correct={cor} "
                      f"({time.time() - t0:.0f}s)", flush=True)

    # ----------------------------------------------------------------
    # Final 3x3 summary.
    # ----------------------------------------------------------------
    print(f"\nSaved {out_npz}")
    print()
    print("# 3x3 medians (GRUB stopping time)")
    rho_K_label = 'rho \\ K'
    header = f"{rho_K_label:>10}  " + "  ".join(
        f"{K:>12d}" for K in Ks)
    print(header)
    print('-' * len(header))
    for ri, rho in enumerate(rhos):
        row = f"{rho:>10.1f}  "
        cells = []
        for ki, K in enumerate(Ks):
            ts = stop_times[ri, ki, :]
            n_cap = int(np.sum(ts >= args.max_steps))
            n_unconv = int(np.sum(~converged[ri, ki, :] & done[ri, ki, :]))
            n_wrong = int(np.sum(~correct[ri, ki, :] & done[ri, ki, :]))
            med = np.nanmedian(ts) if not np.all(np.isnan(ts)) else float('nan')
            tag = ""
            if n_cap > 0:
                tag = f" (cap:{n_cap})"
            elif n_unconv > 0:
                tag = f" (unconv:{n_unconv})"
            if n_wrong > 0:
                tag += f" (wrong:{n_wrong})"
            cells.append(f"{med:>12,.0f}{tag}")
        row += "  ".join(cells)
        print(row)

    # Slope check: log(t)/log(K) per rho.
    print()
    print("# slope log(t_med)/log(K) per rho (linear ~ 1.0, "
          "constant in K ~ 0.0)")
    for ri, rho in enumerate(rhos):
        meds = [np.nanmedian(stop_times[ri, ki, :]) for ki in range(n_K)]
        if any(np.isnan(m) for m in meds):
            print(f"  rho={rho}: incomplete row, skipping slope")
            continue
        log_t = np.log(meds)
        log_K = np.log(Ks)
        slope = float(np.polyfit(log_K, log_t, 1)[0])
        print(f"  rho={rho:6.1f}: slope={slope:+.3f}  meds={meds}")


if __name__ == "__main__":
    main()
