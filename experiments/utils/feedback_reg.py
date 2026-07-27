"""Task 2 -- graph-feedback BAI with a Laplacian-regularized estimator.

One parameterized algorithm covering the full 2x2x2 design, so that the
regularized and empirical arms of the comparison differ *only* in the
estimator (no incidental protocol drift between separately-written classes):

  estimator : 'emp' -- unbiased empirical mean over side observations,
                       mu_hat_i = R_i^fb / N_i^fb,  t_eff,i = N_i^fb
              'reg' -- Laplacian-regularized least squares over side
                       observations,
                         V_t = sum_{s<=t} sum_{j in N+(pi_s)} e_j e_j^T
                               + rho L_G + rho_diag I
                         mu_hat = V_t^{-1} sum_{s<=t} sum_{j in N+(pi_s)}
                                  e_j r_{s,j}
                         t_eff,i = 1 / [V_t^{-1}]_ii
  stop      : 'ts'  -- agreement over M(delta,q,t) Thompson copies
              'ucb' -- UCB-LCB elimination until one arm remains
  pull      : 'cover' -- argmax_a |N+(a) ∩ {i_hat, i_tilde}|
              'width' -- argmax confidence width

The hypothesis under test (from paper/plan.md Task 2): that the TS advantage
comes from the agreement stopping rule being insensitive to an inflated
confidence radius, so that giving the feedback setting a *biased* estimator
should change the TS-vs-UCB ordering.  With 'emp' the estimator is unbiased;
with 'reg' it is Laplacian-biased and the UCB radius carries a sqrt(rho)*eps
bias term.  Measurements are in REBUTTAL_FINDINGS.md section 7.

The UCB radius uses the sqrt(rho)*eps bias rather than rho*eps.  rho*eps
would give a wider UCB radius; sqrt(rho)*eps is the narrower of the two.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm

import support_func
from algobase import build_kernel


def greedy_dominating_set(Adj):
    A = np.asarray(Adj, dtype=float)
    n = A.shape[0]
    closed = (A + np.eye(n)) > 0
    covered = np.zeros(n, dtype=bool)
    dom = []
    while not covered.all():
        gain = np.array([(closed[a] & ~covered).sum() for a in range(n)])
        a = int(np.argmax(gain))
        dom.append(a)
        covered |= closed[a]
    return dom


class FeedbackGraphBAI:
    def __init__(self, D, A, mu, delta, q=0.1, sigma=1.0,
                 estimator='emp', stop='ts', pull='cover',
                 rho=0.0, eps_nominal=None, rho_diag=1e-4,
                 kernel='combinatorial'):
        if estimator not in ('emp', 'reg'):
            raise ValueError(estimator)
        if stop not in ('ts', 'ucb'):
            raise ValueError(stop)
        if pull not in ('cover', 'width'):
            raise ValueError(pull)
        self.means = np.asarray(mu, dtype=float).flatten()
        self.Adj = np.asarray(A, dtype=float)
        self.K = len(self.means)
        self.delta = float(delta)
        self.q = float(q)
        self.sigma = float(sigma)
        self.estimator = estimator
        self.stop = stop
        self.pull = pull
        self.rho = float(rho)
        self.rho_diag = float(rho_diag)
        self.converged = False
        self.t = 0
        self.last_pair = (-1, -1)
        self.last_pull = -1

        self.closed = (self.Adj + np.eye(self.K)) > 0
        self.N_fb = np.zeros(self.K)
        self.R_fb = np.zeros(self.K)
        self.pull_counts = np.zeros(self.K)
        self.remaining_nodes = list(range(self.K))

        self.L = build_kernel(D, A, kernel=kernel)
        # realized smoothness of the instance, used only in the UCB radius
        self.eps = (float(np.sqrt(max(self.means @ self.L @ self.means, 0.0)))
                    if eps_nominal is None else float(eps_nominal))

        if self.estimator == 'reg':
            self.L_rho = self.rho * self.L + self.rho_diag * np.eye(self.K)
            self.inverse_tracker = np.linalg.inv(self.L_rho)
        else:
            self.inverse_tracker = None

        self.sigma0 = 2.0 * self.sigma * np.sqrt(14.0)
        for a in greedy_dominating_set(self.Adj):
            self._pull(a)

    # -- data ----------------------------------------------------------

    def _pull(self, a):
        a = int(a)
        nbrs = np.where(self.closed[a])[0]
        for j in nbrs:
            r = self.means[j] + self.sigma * np.random.randn()
            self.N_fb[j] += 1
            self.R_fb[j] += r
            if self.estimator == 'reg':
                e = np.zeros(self.K)
                e[j] = 1.0
                self.inverse_tracker = support_func.sherman_morrison_inverse(
                    e, self.inverse_tracker)
        self.pull_counts[a] += 1
        self.t += 1

    # -- estimator -----------------------------------------------------

    def _mu_hat(self):
        if self.estimator == 'emp':
            return self.R_fb / np.maximum(self.N_fb, 1.0)
        return np.asarray(self.inverse_tracker @ self.R_fb).flatten()

    def _teff(self):
        if self.estimator == 'emp':
            return np.maximum(self.N_fb, 1.0)
        d = np.diag(self.inverse_tracker)
        return 1.0 / np.maximum(d, 1e-300)

    def _L1(self):
        t_safe = max(float(self.t), 1.0)
        return np.log(max(12.0 * self.K ** 2 * t_safe ** 2 / self.delta, 2.0))

    def _radius(self):
        """UCB-LCB confidence radius.

        The noise term is IDENTICAL in both estimators -- sigma*sqrt(2 L1) --
        and 'reg' only adds the Laplacian bias sqrt(rho)*eps.  This matters:
        an earlier version used GRUB's noise constant sigma_0 = 2 sigma sqrt(14)
        for 'reg' and sigma*sqrt(2) for 'emp', a 5.3x radius gap (~28x in
        sample complexity) that confounded the estimator comparison with a
        constant-factor difference.  Keeping the noise term matched means any
        TS-vs-UCB flip between 'emp' and 'reg' is attributable to the bias
        alone, which is the whole point of the experiment.
        """
        teff = self._teff()
        noise = self.sigma * np.sqrt(2.0 * self._L1())
        if self.estimator == 'emp':
            return noise / np.sqrt(teff)
        bias = np.sqrt(self.rho) * self.eps
        return (noise + bias) / np.sqrt(teff)

    def _variance_factor(self):
        phi_q = norm.isf(self.q)
        return (self.sigma ** 2) * self._L1() / (phi_q ** 2)

    def _n_copies(self):
        return max(int(np.floor(max(self._L1(), 1.0) / self.q)), 1)

    # -- one round -----------------------------------------------------

    def play_round(self, n_rounds=1):
        if self.converged:
            return self.remaining_nodes[0]
        mu_hat = self._mu_hat()
        teff = self._teff()

        if self.stop == 'ts':
            i_hat = int(np.argmax(mu_hat))
            sd = np.sqrt(np.maximum(self._variance_factor() / teff, 0.0))
            thetas = mu_hat + sd * np.random.randn(self._n_copies(), self.K)
            i_tilde_m = np.argmax(thetas, axis=1)
            if np.all(i_tilde_m == i_hat):
                self.converged = True
                self.remaining_nodes = [i_hat]
                self.last_pair = (i_hat, i_hat)
                return i_hat
            m_star = int(np.argmax(np.max(thetas - mu_hat, axis=1)))
            i_tilde = int(i_tilde_m[m_star])
            cand = [i_hat, i_tilde]
        else:
            beta = self._radius()
            upper, lower = mu_hat + beta, mu_hat - beta
            rem = np.array(self.remaining_nodes, dtype=int)
            max_lower = float(np.max(lower[rem]))
            self.remaining_nodes = [i for i in self.remaining_nodes
                                    if upper[i] >= max_lower]
            if len(self.remaining_nodes) <= 1:
                self.converged = True
                return (self.remaining_nodes[0] if self.remaining_nodes
                        else -1)
            rem = np.array(self.remaining_nodes, dtype=int)
            h_star = int(rem[int(np.argmax(mu_hat[rem]))])
            ucb = upper.copy()
            mask = np.ones(self.K, dtype=bool)
            mask[rem] = False
            ucb[mask] = -np.inf
            ucb[h_star] = -np.inf
            l_star = int(np.argmax(ucb))
            i_hat, i_tilde = h_star, l_star
            cand = [h_star, l_star]

        self.last_pair = (int(i_hat), int(i_tilde))

        if self.pull == 'cover':
            best_a, best_key = -1, None
            for a in range(self.K):
                cover = int(self.closed[a, cand].sum())
                key = (-cover, self.N_fb[a])
                if best_key is None or key < best_key:
                    best_key, best_a = key, a
            a_star = best_a
        else:
            beta = self._radius()
            if self.stop == 'ucb':
                rem = np.array(self.remaining_nodes, dtype=int)
                a_star = int(rem[int(np.argmax(beta[rem]))])
            else:
                a_star = int(np.lexsort((self.N_fb, -beta))[0])

        self.last_pull = int(a_star)
        self._pull(a_star)
        return None
