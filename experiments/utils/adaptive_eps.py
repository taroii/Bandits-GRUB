"""Residual-based smoothness estimator with a doubling schedule (Task 3).

A practitioner does not know the true smoothness eps_true = ||mu||_G.  This
module implements the deliverable from Task 3: certify a *lower* bound on
eps from the data, and double the nominal budget eps_bar whenever the
certified lower bound exceeds it (which proves eps_bar was under-specified).

Estimator.  eps_hat(t) = sqrt(<mu_hat(t), L_G mu_hat(t)>).  Since mu_hat is
noisy and Laplacian-biased, eps_hat is not itself a bound on eps.  We
certify a lower bound by subtracting a deviation term: writing
eps = ||L^{1/2} mu|| and using the reverse triangle inequality,

    eps  >=  eps_hat(t)  -  ||L^{1/2}(mu_hat - mu)||    =:  eps_lo(t).

The default bounds the inflation in the V-norm rather than coordinatewise.
Using V_t >= rho L_G, i.e. L_G <= V_t / rho:

    ||L^{1/2}(mu_hat - mu)||^2 = (mu_hat-mu)^T L (mu_hat-mu)
                              <= ||mu_hat - mu||_V^2 / rho

and decomposing mu_hat - mu = V^{-1}S_t - rho V^{-1} L mu,

    ||mu_hat - mu||_V <= sqrt(S_t^T V^{-1} S_t) + rho sqrt(mu^T L V^{-1} L mu)
                      <= sigma sqrt(2 K L1(t))    + sqrt(rho) * eps_bar

(the second term again by V >= rho L).  Hence

    infl(t) = sigma sqrt(2 K L1(t) / rho)  +  eps_bar,

so the test "eps_lo > eps_bar" is equivalent to
eps_hat - sigma sqrt(2 K L1/rho) > 2 eps_bar.  Note infl decreases as rho
grows.

An alternative bound on ||L^{1/2}(mu_hat-mu)|| is
sqrt(lambda_max(L)) * ||c(t)||_2 with coordinatewise radii
c_i = (sigma_0 sqrt(L1) + sqrt(rho) eps_bar) / sqrt(t_eff,i).  Measured on the
K=31 SBM at eps_bar = eps_true/8 it gives infl = 18.36, versus 0.1061 for the
V-norm bound above, against eps_hat = 0.0019.  Both are selectable via
`inflation=`; measured doubling counts under each are in
REBUTTAL_FINDINGS.md section 9.

Doubling.  When eps_lo(t) > eps_bar, set eps_bar <- 2*eps_bar and
rho <- rho_var(eps_bar).  No data is discarded: V_t is rebuilt from the
existing counter and reward sums at the new rho.
"""
from __future__ import annotations

import numpy as np

import graph_algo
from experiments.utils import hardness


class AdaptiveEpsilonTS(graph_algo.ThompsonSampling):
    """TS-Explore that doubles its nominal eps when the data refutes it.

    Extra parameters
    ----------------
    eps_bar0      initial nominal smoothness budget
    T_estimate    horizon estimate used inside rho_var / L1
    check_every   how often (in rounds) to run the certificate
    rho_diag      kept FIXED (not scaled with rho) so that the ridge term
                  does not hand out free effective samples; see
                  hardness.rho_var's docstring
    max_doublings safety cap
    """

    def __init__(self, D, A, mu, delta, q, eps_bar0, T_estimate,
                 check_every=500, rho_diag=1e-4, max_doublings=30,
                 sigma=1.0, reward_fn=None, inflation='vnorm',
                 residual='probe', rho_probe=1.0):
        if inflation not in ('vnorm', 'coordwise'):
            raise ValueError(inflation)
        if residual not in ('plugin', 'probe'):
            raise ValueError(residual)
        self.inflation = inflation
        self.residual = residual
        self.rho_probe = float(rho_probe)
        self.eps_bar = float(eps_bar0)
        self._T_estimate = float(T_estimate)
        self._sigma = float(sigma)
        self._check_every = int(check_every)
        self._max_doublings = int(max_doublings)
        self.n_doublings = 0
        self.doubling_log = []          # (t, eps_bar_before, eps_lo, rho_after)
        K = len(np.asarray(mu).flatten())
        rho0 = hardness.rho_var(self.eps_bar, K, T_estimate, delta,
                                sigma=sigma)
        super().__init__(D=D, A=A, mu=mu, rho_lap=rho0, delta=delta, q=q,
                         epsilon_nominal=self.eps_bar, rho_diag=rho_diag,
                         reward_fn=reward_fn)
        self._lam_max = float(np.linalg.eigvalsh(self.L)[-1])
        self._sigma0 = 2.0 * self._sigma * np.sqrt(14.0)

    # -- certificate ----------------------------------------------------

    def _eps_hat_probe(self):
        """Debiased residual from a PROBE estimator at fixed small rho_probe.

        The plug-in residual <mu_hat, L mu_hat> is computed from the operating
        estimator, whose entire job is to make mu_hat smooth on G.  So it
        collapses toward 0 exactly as rho grows -- measured on the K=31 SBM:

            rho      1     1e2     1e4    7e5
            eps_hat  0.975  0.434  0.111  0.0019      (eps_true = 0.412)

        Under-specification raises rho.  Measured doubling counts under the
        plug-in and probe estimators are in REBUTTAL_FINDINGS.md section 9.

        Instead we re-solve at a fixed rho_probe (decoupled from the operating
        rho) using the same data, and subtract the noise contribution:

            V_p = N + rho_probe L + rho_diag I,   mu_p = V_p^{-1} R
            E<mu_p, L mu_p> = <E mu_p, L E mu_p> + tr(L Cov(mu_p)),
            Cov(mu_p) = sigma^2 V_p^{-1} N V_p^{-1}

        so  eps_hat^2 = <mu_p, L mu_p> - sigma^2 tr(L V_p^{-1} N V_p^{-1}),
        clipped at 0.  rho_probe is decoupled from the operating rho.
        """
        N = self.counter
        V_p = N + self.rho_probe * self.L + self.rho_diag * np.eye(self.dim)
        try:
            V_pi = np.linalg.inv(V_p)
        except np.linalg.LinAlgError:
            return 0.0
        mu_p = V_pi @ self.total_reward
        quad = float(mu_p @ self.L @ mu_p)
        noise = (self._sigma ** 2) * float(np.trace(self.L @ V_pi @ N @ V_pi))
        return float(np.sqrt(max(quad - noise, 0.0)))

    def _eps_lower_bound(self, t):
        self.estimate_mean()
        mu_hat = np.asarray(self.mean_estimate).flatten()
        if self.residual == 'plugin':
            eps_hat = float(np.sqrt(max(mu_hat @ self.L @ mu_hat, 0.0)))
        else:
            eps_hat = self._eps_hat_probe()
        L1 = np.log(12.0 * self.K ** 2 * max(t, 1.0) ** 2 / self.delta)
        if self.inflation == 'vnorm':
            # infl = sigma sqrt(2 K L1 / rho) + eps_bar   (see module docstring)
            infl = (self._sigma * np.sqrt(2.0 * self.K * L1
                                          / max(self.rho_lap, 1e-300))
                    + self.eps_bar)
        else:
            teff = self.get_all_teff()
            c = ((self._sigma0 * np.sqrt(L1)
                  + np.sqrt(self.rho_lap) * self.eps_bar)
                 / np.sqrt(np.maximum(teff, 1e-300)))
            infl = np.sqrt(self._lam_max) * float(np.linalg.norm(c))
        return max(eps_hat - infl, 0.0), eps_hat, infl

    def _rebuild_at(self, rho_new):
        """Re-solve with a new rho, keeping all data."""
        self.rho_lap = float(rho_new)
        self.L_rho = self.rho_lap * self.L + self.rho_diag * np.identity(self.dim)
        self.inverse_tracker = np.linalg.inv(self.counter + self.L_rho)
        self.update_conf_width()
        self._half_rho_L_mu = 0.5 * self.rho_lap * np.dot(self.L, self.means)
        self.eps = self.eps_bar

    def maybe_double(self):
        t = float(np.trace(self.counter))
        if self._check_every <= 0 or (int(t) % self._check_every) != 0:
            return
        if self.n_doublings >= self._max_doublings:
            return
        eps_lo, eps_hat, infl = self._eps_lower_bound(t)
        while (eps_lo > self.eps_bar
               and self.n_doublings < self._max_doublings):
            before = self.eps_bar
            self.eps_bar *= 2.0
            rho_new = hardness.rho_var(self.eps_bar, self.K,
                                       self._T_estimate, self.delta,
                                       sigma=self._sigma)
            self._rebuild_at(rho_new)
            self.n_doublings += 1
            self.doubling_log.append((int(t), before, float(eps_lo),
                                      float(rho_new)))
            eps_lo, eps_hat, infl = self._eps_lower_bound(t)

    def play_round(self, n_rounds=1):
        out = super().play_round(n_rounds)
        if out is None:
            self.maybe_double()
        return out
