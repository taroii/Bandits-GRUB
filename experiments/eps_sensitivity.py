"""Task 3 -- epsilon over/under-specification sweep (Reviewer 4Y4U's Q5).

Runs TS-Explore with a NOMINAL smoothness budget eps_bar while the instance's
TRUE smoothness is eps_true, sweeping the ratio eps_bar/eps_true over a log
grid, with rho set as a practitioner who believed eps_bar would set it:

    rho = rho_var(eps_bar) = sigma_0^2 L1(T) / eps_bar^2      (Eq. (5))

Under-specification (ratio < 1) sets rho larger; over-specification (ratio > 1)
sets it smaller.  Both stopping time and correctness are recorded, since
Reviewer 4Y4U's Q5 concerns whether under-specification affects correctness and
not only speed.

Two rho_diag policies are available, and the difference matters:
  fixed  (default) rho_diag = 1e-4 regardless of rho.
  scaled           rho_diag = max(1e-4, 1e-6 rho), the Appendix H.1 policy.
Under either policy the pre-pull effective sample size t_eff,i(0) equals
K*rho_diag exactly (verified); the scaled policy makes that grow with rho.
--rho-diag-policy runs both.

Also runs the adaptive variant (residual estimator + doubling schedule) and
reports the number of doublings and its multiplicative cost vs. oracle eps.

  python experiments/eps_sensitivity.py --seeds 5
  python experiments/eps_sensitivity.py --seeds 5 --adaptive
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.utils import instances, hardness, runners  # noqa: E402
from experiments.utils.adaptive_eps import AdaptiveEpsilonTS  # noqa: E402
import graph_algo  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs')
os.makedirs(OUT, exist_ok=True)

RATIOS = [0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, default=5)
    ap.add_argument('--q', type=float, default=0.1)
    ap.add_argument('--ratios', type=float, nargs='+', default=RATIOS)
    ap.add_argument('--max-steps', type=int, default=10_000_000)
    ap.add_argument('--rho-diag-policy', choices=['fixed', 'scaled'],
                    default='fixed')
    ap.add_argument('--adaptive', action='store_true',
                    help="also run the residual-estimator + doubling variant")
    ap.add_argument('--check-every', type=int, default=500,
                    help="adaptive variant: certificate interval")
    ap.add_argument('--residual', choices=['probe', 'plugin'], default='probe',
                    help="'plugin' = eps_hat^2 = <mu_hat, L mu_hat> from the "
                         "operating estimator; 'probe' = debiased re-solve at "
                         "a fixed rho_probe. Measured behaviour of both is in "
                         "REBUTTAL_FINDINGS.md section 9.")
    ap.add_argument('--rho-probe', type=float, default=1.0)
    ap.add_argument('--fresh', action='store_true')
    args = ap.parse_args()

    delta = 1e-3
    seeds = list(range(args.seeds))
    tag = f'eps_sensitivity_{args.rho_diag_policy}'
    out_npz = os.path.join(OUT, f'{tag}_results.npz')

    mu, A, D = instances.sbm_phase_transition_connected(seed=0)
    K = len(mu)
    L = D - A
    eps_true = float(np.sqrt(max(mu @ L @ mu, 0.0)))
    a_star = int(np.argmax(mu))
    gaps = (mu[a_star] - mu)[mu < mu[a_star]]
    H_cls = hardness.classical_hardness(mu)
    T_est = H_cls * np.log(1.0 / delta)
    lam_max = float(np.linalg.eigvalsh(L)[-1])

    print(f"[{tag}] connected SBM K={K} edges={int(A.sum()/2)}")
    print(f"  eps_true={eps_true:.4f}  H_classical={H_cls:.2f}  "
          f"sum 1/D^2={ (1/gaps**2).sum():.2f}  max 1/D^2={(1/gaps**2).max():.2f}")
    print(f"  T_est={T_est:.0f}  rho_diag policy={args.rho_diag_policy}")
    print(f"  NOTE rho is set from Eq.(5) rho_var(eps_bar), not "
          f"hardness.rho_star (which is sqrt of it; see its docstring)",
          flush=True)

    ratios = list(args.ratios)
    n = len(ratios)
    stop = np.full((n, len(seeds)), np.nan)
    correct = np.zeros((n, len(seeds)), dtype=bool)
    capped = np.zeros((n, len(seeds)), dtype=bool)
    rhos = np.full(n, np.nan)
    free_teff = np.full(n, np.nan)
    done = np.zeros(n, dtype=bool)

    def save():
        tmp = out_npz + '.tmp.npz'
        np.savez(tmp, ratios=np.array(ratios), seeds=np.array(seeds),
                 stop=stop, correct=correct.astype(int),
                 capped=capped.astype(int), rhos=rhos, free_teff=free_teff,
                 done=done.astype(int), eps_true=eps_true, K=K,
                 delta=delta, q=args.q, T_est=T_est,
                 rho_diag_policy=np.array(args.rho_diag_policy),
                 max_steps=int(args.max_steps))
        os.replace(tmp, out_npz)

    if os.path.exists(out_npz) and not args.fresh:
        try:
            z = np.load(out_npz, allow_pickle=False)
            if (list(z['ratios'].tolist()) == ratios
                    and list(z['seeds'].tolist()) == seeds
                    and str(z['rho_diag_policy']) == args.rho_diag_policy):
                stop, correct = z['stop'], z['correct'].astype(bool)
                capped, done = z['capped'].astype(bool), z['done'].astype(bool)
                rhos, free_teff = z['rhos'], z['free_teff']
                print(f"[resume] {int(done.sum())}/{n} cells", flush=True)
        except Exception as e:
            print(f"[resume] failed: {e}", flush=True)

    print(f"\n{'ratio':>7} {'eps_bar':>9} {'rho':>11} {'rho_diag':>9} "
          f"{'free_teff':>10} {'t_med':>11} {'correct':>8} {'cap':>5}",
          flush=True)

    for i, r in enumerate(ratios):
        eps_bar = r * eps_true
        rho = hardness.rho_var(eps_bar, K, T_est, delta)
        rho_diag = (1e-4 if args.rho_diag_policy == 'fixed'
                    else max(1e-4, 1e-6 * rho))
        rhos[i] = rho
        free_teff[i] = K * rho_diag
        if done[i]:
            print(f"{r:>7.3f} {eps_bar:>9.4f} {rho:>11.4g} {rho_diag:>9.2e} "
                  f"{free_teff[i]:>10.2e} {np.nanmedian(stop[i]):>11,.0f} "
                  f"{correct[i].mean():>7.0%} {capped[i].sum():>3d}/"
                  f"{len(seeds)}  [resumed]", flush=True)
            continue

        def fac(rho=rho, eps_bar=eps_bar, rho_diag=rho_diag):
            return graph_algo.ThompsonSampling(
                D=D, A=A, mu=mu, rho_lap=rho, delta=delta, q=args.q,
                epsilon_nominal=eps_bar, rho_diag=rho_diag)

        t0 = time.time()
        runs = runners.run_many(fac, seeds, max_steps=args.max_steps,
                               record_elimination=False)
        stop[i] = [x['stopping_time'] for x in runs]
        correct[i] = [x['correct'] for x in runs]
        capped[i] = [not x['converged_flag'] for x in runs]
        done[i] = True
        save()
        print(f"{r:>7.3f} {eps_bar:>9.4f} {rho:>11.4g} {rho_diag:>9.2e} "
              f"{free_teff[i]:>10.2e} {np.median(stop[i]):>11,.0f} "
              f"{correct[i].mean():>7.0%} {capped[i].sum():>3d}/{len(seeds)}  "
              f"({time.time()-t0:.0f}s)", flush=True)

    save()

    # ---- summary -----------------------------------------------------
    print(f"\n# Summary (eps_true = {eps_true:.4f})")
    i1 = ratios.index(1.0) if 1.0 in ratios else None
    base = np.nanmedian(stop[i1]) if i1 is not None else np.nan
    print(f"{'ratio':>7} {'regime':>18} {'t_med':>11} {'vs ratio=1':>11} "
          f"{'correct':>8}")
    for i, r in enumerate(ratios):
        regime = ('under-specified' if r < 1 else
                  'oracle' if r == 1 else 'over-specified')
        m = np.nanmedian(stop[i])
        print(f"{r:>7.3f} {regime:>18} {m:>11,.0f} "
              f"{(m/base if base==base else np.nan):>11.2f} "
              f"{correct[i].mean():>7.0%}")
    under = [i for i, r in enumerate(ratios) if r < 1]
    if under:
        cr = np.concatenate([correct[i] for i in under])
        print(f"\n  correctness over ALL under-specified cells: "
              f"{cr.mean():.1%} ({cr.sum()}/{len(cr)} runs returned a*)")
        if cr.all():
            print("  -> correctness was 1.0 in every under-specified cell run.")

    # ---- adaptive variant --------------------------------------------
    ad_ratios = [0.125, 0.25, 0.5, 1.0]
    ad_stop = np.full((len(ad_ratios), len(seeds)), np.nan)
    ad_cor = np.zeros((len(ad_ratios), len(seeds)), dtype=bool)
    ad_ndoub = np.full((len(ad_ratios), len(seeds)), np.nan)
    ad_epsend = np.full((len(ad_ratios), len(seeds)), np.nan)

    def save_adaptive():
        f = os.path.join(OUT, f'{tag}_adaptive_{args.residual}_results.npz')
        tmp = f + '.tmp.npz'
        np.savez(tmp, ratios0=np.array(ad_ratios), seeds=np.array(seeds),
                 stop=ad_stop, correct=ad_cor.astype(int),
                 n_doublings=ad_ndoub, eps_bar_end=ad_epsend,
                 eps_true=eps_true, residual=np.array(args.residual),
                 rho_probe=float(args.rho_probe),
                 check_every=int(args.check_every),
                 baseline_T=float(base) if base == base else np.nan,
                 K=K, delta=delta, q=args.q, T_est=T_est,
                 max_steps=int(args.max_steps))
        os.replace(tmp, f)
        return f

    if args.adaptive:
        print(f"\n# Adaptive (residual estimator eps_hat^2 = <mu_hat, L mu_hat> "
              f"+ doubling)")
        print(f"  lambda_max(L) = {lam_max:.3f}")
        print(f"  residual mode = {args.residual}"
              + (f" (rho_probe={args.rho_probe:g})"
                 if args.residual == 'probe' else ""))
        print(f"{'eps_bar0':>9} {'ratio0':>7} {'t_med':>11} {'correct':>8} "
              f"{'doublings':>10} {'eps_end/eps_true':>17} "
              f"{'cost vs oracle':>15}")
        for r0i, r0 in enumerate(ad_ratios):
            eps0 = r0 * eps_true
            ts, cors, nds, ends = [], [], [], []
            for seed in seeds:
                def fac(eps0=eps0):
                    return AdaptiveEpsilonTS(
                        D=D, A=A, mu=mu, delta=delta, q=args.q,
                        eps_bar0=eps0, T_estimate=T_est,
                        check_every=args.check_every, rho_diag=1e-4,
                        residual=args.residual,
                        rho_probe=args.rho_probe)
                np.random.seed(seed)
                algo = fac()
                steps = 0
                while (not getattr(algo, 'converged', False)
                       and steps < args.max_steps):
                    algo.play_round(1)
                    steps += 1
                ts.append(float(np.trace(algo.counter)))
                cors.append(int(algo.remaining_nodes[0]) == a_star)
                nds.append(algo.n_doublings)
                ends.append(algo.eps_bar)
            ad_stop[r0i] = ts
            ad_cor[r0i] = cors
            ad_ndoub[r0i] = nds
            ad_epsend[r0i] = ends
            save_adaptive()
            print(f"{eps0:>9.4f} {r0:>7.3f} {np.median(ts):>11,.0f} "
                  f"{np.mean(cors):>7.0%} {np.median(nds):>10.1f} "
                  f"{np.median(ends)/eps_true:>17.3f} "
                  f"{(np.median(ts)/base if base==base else np.nan):>15.2f}")
        print(f"\n  (cost vs oracle = median T / median T at ratio=1)")
        print(f"  Saved {save_adaptive()}")

    print(f"\nSaved {out_npz}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
