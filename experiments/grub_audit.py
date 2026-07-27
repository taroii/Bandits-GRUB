"""Task 0 -- GRUB confidence-radius audit (diagnostic, no bandit runs).

Decomposes the GRUB elimination radius into its noise and bias parts and
compares the four candidate bias terms:

  published   rho * eps * sqrt([V^-1]_ii)          (Thaker et al. 2022, Lemma 2 /
                                                    Alg. 1 line: beta = 2 sigma
                                                    sqrt(14 log(2n(t+1)^2/delta))
                                                    + rho*eps)
  sqrt        sqrt(rho) * eps * sqrt([V^-1]_ii)    (tight bound implied by
                                                    V_t >= rho L_G, i.e. our
                                                    own Section 3.2 argument)
  oracle      |[V^-1 rho L_G mu]_i|                (exact bias, needs true mu)
  legacy      0.5 * [V^-1 rho L_G mu]_i, added to
              the LOWER bound only                 (what algobase.py actually
                                                    does today)

and the noise multiplier under natural log vs log2 (algobase.py uses np.log2).

Run:  python experiments/grub_audit.py
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.utils import instances  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs')
os.makedirs(OUT, exist_ok=True)


def conc_radius(K, t, delta, base):
    """GRUB noise multiplier 2*sigma*sqrt(14*log(2Kt/delta)), sigma=1.

    ``base`` selects the logarithm: 'ln' is the published expression,
    'log2' is what algobase.py currently computes.
    """
    arg = max(2.0 * K * t / delta, 2.0)
    lg = np.log2(arg) if base == 'log2' else np.log(arg)
    return 2.0 * np.sqrt(14.0 * max(lg, 0.0))


def bias_terms(V_inv, L, mu, rho, eps):
    """Return dict of per-arm bias magnitudes under each convention."""
    d = np.sqrt(np.maximum(np.diag(V_inv), 0.0))          # sqrt([V^-1]_ii)
    exact_signed = V_inv @ (rho * L @ mu)                  # [V^-1 rho L mu]_i
    return {
        'published': rho * eps * d,
        'sqrt': np.sqrt(rho) * eps * d,
        'oracle': np.abs(exact_signed),
        'legacy_signed': 0.5 * exact_signed,
        'sqrt_Vinv_ii': d,
    }


def uniform_V_inv(L, rho, rho_diag, t_per_arm, K):
    """V_t under a uniform allocation of t_per_arm pulls to every arm."""
    V = t_per_arm * np.eye(K) + rho * L + rho_diag * np.eye(K)
    return np.linalg.inv(V)


def audit_instance(name, mu, A, D, rho, delta, t_grid, rho_diag=1e-4):
    K = len(mu)
    L = D - A
    eps = float(np.sqrt(max(mu @ L @ mu, 0.0)))
    gaps = mu.max() - mu
    gaps_nz = gaps[gaps > 0]
    print(f"\n{'='*78}")
    print(f"{name}:  K={K}  rho={rho:g}  eps={eps:.4f}  "
          f"Delta_min={gaps_nz.min():.4f}  Delta_max={gaps.max():.4f}")
    print(f"{'='*78}")

    # --- noise multiplier: published (ln) vs implemented (log2) -----------
    print("\n  noise multiplier 2*sqrt(14*log(2Kt/delta)):")
    print(f"    {'t':>10}  {'ln (published)':>15}  {'log2 (code)':>12}  "
          f"{'ratio':>7}  {'ratio^2':>8}")
    for t in t_grid:
        a = conc_radius(K, t, delta, 'ln')
        b = conc_radius(K, t, delta, 'log2')
        print(f"    {t:>10,}  {a:>15.3f}  {b:>12.3f}  {b/a:>7.3f}  "
              f"{(b/a)**2:>8.3f}")

    # --- bias vs noise, at a uniform allocation --------------------------
    print("\n  radius decomposition at uniform allocation (per-arm pulls n):")
    print(f"    {'n/arm':>9}  {'noise(ln)':>10}  {'bias_pub':>9}  "
          f"{'bias_sqrt':>10}  {'bias_orac':>10}  {'pub/noise':>10}  "
          f"{'(1+p/n)^2':>10}")
    for t_per_arm in [10, 100, 1_000, 10_000, 100_000]:
        V_inv = uniform_V_inv(L, rho, rho_diag, t_per_arm, K)
        t_total = t_per_arm * K
        bt = bias_terms(V_inv, L, mu, rho, eps)
        d = bt['sqrt_Vinv_ii']
        noise = conc_radius(K, t_total, delta, 'ln') * d
        # Report the worst-case (max over arms) of each bias.
        pub = bt['published'].max()
        sq = bt['sqrt'].max()
        orc = bt['oracle'].max()
        nz = noise.max()
        infl = (1.0 + pub / nz) ** 2
        print(f"    {t_per_arm:>9,}  {nz:>10.4f}  {pub:>9.4f}  "
              f"{sq:>10.4f}  {orc:>10.5f}  {pub/nz:>10.3f}  {infl:>10.2f}")

    # --- how many pulls to eliminate the hardest challenger -------------
    # Elimination needs radius_i + radius_best < Delta_i.  Approximate with
    # 2 * radius at the bottleneck gap.
    print("\n  approx per-arm effective samples needed to eliminate at "
          f"Delta_min={gaps_nz.min():.3f}")
    print(f"    (solve 2*(beta_noise + beta_bias)/sqrt(t_eff) = Delta_min)")
    t_ref = 1_000_000
    for label, log_base, bias_kind in [
            ('published (rho*eps, ln)', 'ln', 'published'),
            ('sqrt(rho)*eps, ln', 'ln', 'sqrt'),
            ('no bias, ln', 'ln', 'none'),
            ('no bias, log2 (code noise)', 'log2', 'none'),
    ]:
        beta_noise = conc_radius(K, t_ref, delta, log_base)
        if bias_kind == 'published':
            beta_bias = rho * eps
        elif bias_kind == 'sqrt':
            beta_bias = np.sqrt(rho) * eps
        else:
            beta_bias = 0.0
        beta = beta_noise + beta_bias
        t_eff_needed = (2.0 * beta / gaps_nz.min()) ** 2
        print(f"    {label:<28s} beta={beta:8.3f}  "
              f"t_eff needed={t_eff_needed:14,.0f}")

    return dict(K=K, eps=eps, delta_min=float(gaps_nz.min()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rho', type=float, default=100.0)
    ap.add_argument('--delta', type=float, default=1e-3)
    ap.add_argument('--Ks', type=int, nargs='+', default=[10, 50, 200])
    ap.add_argument('--gap-step', type=float, default=0.3)
    ap.add_argument('--C', type=int, default=2)
    ap.add_argument('--movielens', action='store_true',
                    help="also audit the MovieLens K=20 instance")
    args = ap.parse_args()

    t_grid = [1_000, 100_000, 10_000_000]

    print(__doc__)
    for K in args.Ks:
        mu, A, D = instances.clustered_chain(K, C=args.C,
                                             gap_step=args.gap_step)
        audit_instance(f"clustered_chain K={K}", mu, A, D,
                       args.rho, args.delta, t_grid)

    if args.movielens:
        for rho in [1.0, 100.0, 1000.0]:
            mu, A, D = instances.movielens_top_k(K=20, top_k_neighbors=5)
            audit_instance(f"movielens K=20 rho={rho:g}", mu, A, D,
                           rho, args.delta, t_grid)


if __name__ == '__main__':
    main()
