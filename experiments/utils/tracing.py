"""Task 1 -- per-run instrumentation for graph-bandit runs.

Records, for a single run, everything needed to answer Reviewer 4Y4U's Q7
("what graph properties drive the reduction?") from traces rather than
assertion, and to re-aggregate results without rerunning:

  teff          (n_snap, K)  effective sample size t_eff,i(t) per arm
  pulls         (n_snap, K)  direct pull counts t_i
  n_fb          (n_snap, K)  feedback observation counts N_i^fb (feedback
                             algorithms only; equals ``pulls`` otherwise)
  mu_hat        (n_snap, K)  current mean estimate
  remaining     (n_snap, K)  bool: arm still in the candidate set
  comp_H        (n_snap, K)  bool: arm in the competitive set H at this t,
                             under the criterion named in the plan,
                               rho*J(i,G)/2 < 186*C(t)*L2(t)/Delta_{i,c}^2
  comp_H_def6   (n_snap, K)  bool: arm in H_eps under Definition 6,
                               Delta_{i,c}^2 * J(i,G) <= c0 * eps^2
  comp_H_thaker (n_snap, K)  bool: arm in H under the Thaker-style criterion
                               rho*J(i,G) <= 1/Delta_{i,c}^2
                             (the one `hardness.competitive_set` uses)
  snap_t        (n_snap,)    the round index t at each snapshot
  pair          (n_pair, 3)  [t, i_hat, i_tilde] disagreement pair per round
  pulled        (n_pair,)    arm actually pulled that round

plus scalars: stopping_time, capped, correct, selected_arm, K, rho, eps,
delta, q, a_star, J (influence factors to a*), Delta.

The three competitive-set criteria are all logged because the draft uses
different ones in different places (Definition 6 vs. Appendix, vs. what
``hardness.competitive_set`` implements); logging all three lets the
authors pick without a rerun.

Tracing is strictly opt-in and consumes no randomness, so a traced run
follows the identical RNG stream to an untraced one with the same seed.
"""
from __future__ import annotations

import os

import numpy as np

from experiments.utils import hardness


def _phi_q_isf(q):
    from scipy.stats import norm
    return float(norm.isf(q))


class Tracer:
    """Snapshot recorder for a single algorithm run.

    Parameters
    ----------
    algo      the algorithm instance (already constructed)
    every     snapshot the per-arm state every ``every`` rounds
    max_snaps hard cap on snapshots; the interval is doubled whenever the
              cap would be exceeded, so memory stays bounded on 1e7-pull
              runs without the caller having to guess ``every``
    log_pairs record the disagreement pair every round (can be large; set
              False to skip). Pairs are also subsampled at ``every`` when
              ``pair_every`` is None.
    """

    def __init__(self, algo, means, Adj, Degree, rho, eps, delta, q,
                 every=100, max_snaps=20_000, log_pairs=True,
                 pair_every=None):
        self.algo = algo
        self.every = int(every)
        self.max_snaps = int(max_snaps)
        self.log_pairs = bool(log_pairs)
        self.pair_every = int(pair_every) if pair_every else int(every)

        self.means = np.asarray(means, dtype=float).flatten()
        self.K = len(self.means)
        self.rho = float(rho)
        self.eps = float(eps)
        self.delta = float(delta)
        self.q = float(q) if q else float('nan')
        self.a_star = int(np.argmax(self.means))
        self.Delta = self.means[self.a_star] - self.means

        # Influence factor J(i, G) to the optimal arm (resistance distance).
        if Adj is not None and np.asarray(Adj).sum() > 0:
            R = hardness.influence_factors(Adj, Degree)
            self.J = R[:, self.a_star].copy()
        else:
            self.J = np.full(self.K, np.inf)

        self._snap_t = []
        self._teff = []
        self._pulls = []
        self._n_fb = []
        self._mu_hat = []
        self._remaining = []
        self._compH = []
        self._compH_def6 = []
        self._compH_thaker = []
        self._pairs = []
        self._pulled = []

    # -- per-algorithm accessors ---------------------------------------

    def _get_teff(self):
        a = self.algo
        if hasattr(a, 'inverse_tracker') and np.size(a.inverse_tracker):
            d = np.diag(a.inverse_tracker)
            return 1.0 / np.maximum(d, 1e-300)
        if hasattr(a, 'N_fb'):
            return np.asarray(a.N_fb, dtype=float).copy()
        if hasattr(a, 'counts'):
            return np.asarray(a.counts, dtype=float).copy()
        return np.full(self.K, np.nan)

    def _get_pulls(self):
        a = self.algo
        if hasattr(a, 'counter') and np.size(getattr(a, 'counter', [])):
            return np.diag(a.counter).copy()
        if hasattr(a, 'pull_counts'):
            return np.asarray(a.pull_counts, dtype=float).copy()
        if hasattr(a, 'counts'):
            return np.asarray(a.counts, dtype=float).copy()
        return np.full(self.K, np.nan)

    def _get_n_fb(self):
        a = self.algo
        if hasattr(a, 'N_fb'):
            return np.asarray(a.N_fb, dtype=float).copy()
        return self._get_pulls()

    def _get_mu_hat(self):
        a = self.algo
        if hasattr(a, 'mean_estimate') and np.size(a.mean_estimate):
            return np.asarray(a.mean_estimate, dtype=float).flatten().copy()
        if hasattr(a, 'R_fb'):
            return (np.asarray(a.R_fb, float)
                    / np.maximum(np.asarray(a.N_fb, float), 1.0))
        if hasattr(a, 'total_reward') and hasattr(a, 'counts'):
            return (np.asarray(a.total_reward, float)
                    / np.maximum(np.asarray(a.counts, float), 1.0))
        return np.full(self.K, np.nan)

    def _get_t(self):
        a = self.algo
        if hasattr(a, 'counter') and np.size(getattr(a, 'counter', [])):
            return int(np.trace(a.counter))
        return int(getattr(a, 't', 0))

    # -- competitive-set criteria --------------------------------------

    def _C_of_t(self, t):
        """Thompson variance factor C(delta, q, t) = L1(t)/phi(q)^2."""
        if not np.isfinite(self.q):
            return np.nan
        t_safe = max(float(t), 1.0)
        L1 = np.log(12.0 * self.K ** 2 * t_safe ** 2 / self.delta)
        phi = _phi_q_isf(self.q)
        return L1 / (phi ** 2)

    def _L2_of_t(self, t):
        """L2(t) = log(12 K^2 t^2 / delta), the union-bound log factor."""
        t_safe = max(float(t), 1.0)
        return np.log(12.0 * self.K ** 2 * t_safe ** 2 / self.delta)

    def _competitive_masks(self, t):
        """Return the three competitive-set indicator vectors at round t."""
        gap2 = self.Delta ** 2
        with np.errstate(divide='ignore', invalid='ignore'):
            # (a) The criterion named in the plan / appendix.
            C_t = self._C_of_t(t)
            L2 = self._L2_of_t(t)
            rhs = 186.0 * C_t * L2 / np.where(gap2 > 0, gap2, np.nan)
            mask_a = (self.rho * self.J / 2.0) < rhs
            # (b) Definition 6 (large-rho regime), c0 = 8.
            mask_b = (gap2 * self.J) <= (8.0 * self.eps ** 2)
            # (c) Thaker-style, as implemented in hardness.competitive_set.
            mask_c = (self.rho * self.J) <= np.where(gap2 > 0,
                                                     1.0 / gap2, np.inf)
        for m in (mask_a, mask_b, mask_c):
            m[~np.isfinite(self.J)] = False
            m[self.a_star] = False
        return mask_a, mask_b, mask_c

    # -- public API -----------------------------------------------------

    def snapshot(self, force=False):
        t = self._get_t()
        if not force and (t % self.every) != 0:
            return
        # Bound memory: halve the resolution once the cap is reached.
        if len(self._snap_t) >= self.max_snaps:
            keep = slice(None, None, 2)
            for lst in (self._snap_t, self._teff, self._pulls, self._n_fb,
                        self._mu_hat, self._remaining, self._compH,
                        self._compH_def6, self._compH_thaker):
                lst[:] = lst[keep]
            self.every *= 2

        rem = np.zeros(self.K, dtype=bool)
        for i in getattr(self.algo, 'remaining_nodes', []):
            rem[int(i)] = True
        ma, mb, mc = self._competitive_masks(t)

        self._snap_t.append(t)
        self._teff.append(self._get_teff())
        self._pulls.append(self._get_pulls())
        self._n_fb.append(self._get_n_fb())
        self._mu_hat.append(self._get_mu_hat())
        self._remaining.append(rem)
        self._compH.append(ma)
        self._compH_def6.append(mb)
        self._compH_thaker.append(mc)

    def record_pair(self):
        """Record the disagreement pair and pulled arm for this round."""
        if not self.log_pairs:
            return
        t = self._get_t()
        if (t % self.pair_every) != 0:
            return
        pair = getattr(self.algo, 'last_pair', None)
        pulled = getattr(self.algo, 'last_pull', -1)
        if pair is None:
            pair = (-1, -1)
        self._pairs.append((t, int(pair[0]), int(pair[1])))
        self._pulled.append(int(pulled if pulled is not None else -1))

    def result(self, stopping_time, selected_arm, correct, capped):
        def _stack(lst, dtype=float):
            if not lst:
                return np.zeros((0, self.K), dtype=dtype)
            return np.asarray(lst, dtype=dtype)

        return dict(
            snap_t=np.asarray(self._snap_t, dtype=np.int64),
            teff=_stack(self._teff),
            pulls=_stack(self._pulls),
            n_fb=_stack(self._n_fb),
            mu_hat=_stack(self._mu_hat),
            remaining=_stack(self._remaining, bool),
            comp_H=_stack(self._compH, bool),
            comp_H_def6=_stack(self._compH_def6, bool),
            comp_H_thaker=_stack(self._compH_thaker, bool),
            pair=(np.asarray(self._pairs, dtype=np.int64)
                  if self._pairs else np.zeros((0, 3), dtype=np.int64)),
            pulled=np.asarray(self._pulled, dtype=np.int64),
            stopping_time=int(stopping_time),
            selected_arm=int(selected_arm),
            correct=bool(correct),
            capped=bool(capped),
            K=int(self.K), rho=float(self.rho), eps=float(self.eps),
            delta=float(self.delta), q=float(self.q),
            a_star=int(self.a_star), J=self.J, Delta=self.Delta,
            means=self.means, snap_every=int(self.every),
        )


def save_trace(trace, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + '.tmp.npz'
    np.savez_compressed(tmp, **trace)
    os.replace(tmp, path)
    return path
