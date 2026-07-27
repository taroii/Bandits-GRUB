"""Task 6 -- TaS-FG (Track-and-Stop for Feedback Graphs) as a live baseline.

Faithful implementation of Russo, Song & Pacchiano, "Pure Exploration with
Feedback Graphs", AISTATS 2025 (arXiv 2503.07824), *informed* case with a
known deterministic feedback graph and Gaussian rewards N(mu_u, lambda^2).

Cost note: the stopping statistic is closed-form for Gaussians and needs no
convex solve.
From their Sec. 3.3, L(t) = t * T(N(t)/t; nu_hat(t))^{-1}, and since
m_u = M_u(t)/t the t cancels:

    L(t) = min_{u != a_hat}  Delta_hat_u(t)^2
                             / ( 2 lambda^2 (1/M_u(t) + 1/M_{a_hat}(t)) )

with M_u(t) = sum_v N_{v,u}(t) the number of times u's reward was observed.
Only the *sampling* rule needs the convex program

    omega^*(t) = arginf_{omega in Delta(V)} max_{u != a_hat}
                     (1/m_u + 1/m_{a_hat}) 2 lambda^2 / Delta_hat_u^2,
                 s.t. m = G^T omega

which we re-solve every ``resolve_every`` rounds rather than every round
(a standard practical relaxation; ``resolve_every=1`` recovers the paper's
algorithm exactly, at ~10ms/round).

Sampling rule is their averaged D-tracking (Proposition 3), the variant they
state converges when omega^* is non-unique (their paper gives a symmetric-graph
example where the optimal set is a convex set):

    S_t = {u : N_u(t) < sqrt(t) - K/2}
    V_t = argmin_{u in S_t} N_u(t)                      if S_t nonempty
          argmin_{u in V} N_u(t) - sum_{n<=t} omega^*_u(n)   otherwise

Threshold (their Eq. (7)):
    beta(t, delta) = 2 C_exp( ln((K-1)/delta) / 2 ) + 6 ln(1 + ln t),
    C_exp(x) ~= x + 4 ln(1 + x + sqrt(2x))     for x >= 5.
Note for K=20, delta=1e-3 the argument is x = 4.93, marginally below the
x >= 5 regime where that approximation is stated; we use it anyway and flag
it (the effect is a fraction of a nat on a threshold of ~41).
"""
from __future__ import annotations

import numpy as np


def c_exp(x):
    """C_exp(x) ~= x + 4 ln(1 + x + sqrt(2x))  (Kaufmann-Koolen Thm 7)."""
    x = max(float(x), 1e-12)
    return x + 4.0 * np.log(1.0 + x + np.sqrt(2.0 * x))


def beta_threshold(t, delta, K):
    t = max(float(t), np.e)
    return (2.0 * c_exp(np.log(max(K - 1, 1) / delta) / 2.0)
            + 6.0 * np.log(1.0 + np.log(t)))


def solve_omega_star(gaps, closed, a_hat, lam=1.0, solver=None):
    """arginf_omega max_{u != a_hat} (1/m_u + 1/m_a) 2 lam^2 / gap_u^2."""
    import cvxpy as cp
    K = len(gaps)
    G = closed.astype(float)          # G[v, u] = 1 iff pulling v reveals u
    sub = [u for u in range(K) if u != a_hat and gaps[u] > 0]
    if not sub:
        return np.ones(K) / K
    w = cp.Variable(K, nonneg=True)
    m = G.T @ w
    tvar = cp.Variable()
    cons = [cp.sum(w) == 1]
    for u in sub:
        cons.append(2.0 * lam ** 2 / gaps[u] ** 2
                    * (cp.inv_pos(m[u]) + cp.inv_pos(m[a_hat])) <= tvar)
    try:
        cp.Problem(cp.Minimize(tvar), cons).solve(
            solver=solver or cp.CLARABEL)
        if w.value is None:
            return np.ones(K) / K
        val = np.maximum(np.asarray(w.value).flatten(), 0.0)
        s = val.sum()
        return val / s if s > 0 else np.ones(K) / K
    except Exception:
        return np.ones(K) / K


class TaSFG:
    """Track-and-Stop for Feedback Graphs (informed, Gaussian)."""

    def __init__(self, D, A, mu, delta, q=None, sigma=1.0,
                 resolve_every=25, solver=None):
        del q  # accepted for factory parity
        self.means = np.asarray(mu, dtype=float).flatten()
        self.Adj = np.asarray(A, dtype=float)
        self.K = len(self.means)
        self.delta = float(delta)
        self.lam = float(sigma)
        self.resolve_every = int(resolve_every)
        self.solver = solver
        self.converged = False
        self.t = 0
        self.last_pair = (-1, -1)
        self.last_pull = -1

        self.closed = (self.Adj + np.eye(self.K)) > 0
        self.M = np.zeros(self.K)        # M_u: times u's reward was observed
        self.R = np.zeros(self.K)        # reward sums over observations
        self.N = np.zeros(self.K)        # N_v: times v was *pulled*
        self.pull_counts = self.N        # alias for the shared runner
        self.remaining_nodes = list(range(self.K))
        self.w_cumsum = np.zeros(self.K)  # sum_{n<=t} omega^*(n)
        self._w_cached = np.ones(self.K) / self.K
        self._n_solves = 0

        # Seed every arm so M_u > 0 and gaps are defined.
        from experiments.utils.feedback_reg import greedy_dominating_set
        for a in greedy_dominating_set(self.Adj):
            self._pull(a)

    def _pull(self, a):
        a = int(a)
        for j in np.where(self.closed[a])[0]:
            self.M[j] += 1
            self.R[j] += self.means[j] + self.lam * np.random.randn()
        self.N[a] += 1
        self.t += 1
        self.last_pull = a

    def _mu_hat(self):
        return self.R / np.maximum(self.M, 1.0)

    def glr(self, mu_hat, a_hat):
        """L(t) = min_{u != a_hat} gap_u^2 / (2 lam^2 (1/M_u + 1/M_a))."""
        best = np.inf
        worst_u = -1
        for u in range(self.K):
            if u == a_hat:
                continue
            gap = mu_hat[a_hat] - mu_hat[u]
            if gap <= 0:
                return 0.0, u
            val = gap ** 2 / (2.0 * self.lam ** 2
                              * (1.0 / max(self.M[u], 1e-12)
                                 + 1.0 / max(self.M[a_hat], 1e-12)))
            if val < best:
                best, worst_u = val, u
        return float(best), int(worst_u)

    def play_round(self, n_rounds=1):
        if self.converged:
            return self.remaining_nodes[0]
        mu_hat = self._mu_hat()
        a_hat = int(np.argmax(mu_hat))
        L, worst_u = self.glr(mu_hat, a_hat)
        self.last_pair = (a_hat, worst_u)

        if L >= beta_threshold(self.t, self.delta, self.K):
            self.converged = True
            self.remaining_nodes = [a_hat]
            return a_hat

        # omega^*(t) on the plug-in instance, re-solved periodically.
        if (self.t % self.resolve_every) == 0 or self._n_solves == 0:
            gaps = np.maximum(mu_hat[a_hat] - mu_hat, 0.0)
            gmin = gaps[gaps > 0].min() if (gaps > 0).any() else 1.0
            gaps = gaps.copy()
            gaps[a_hat] = gmin           # their Delta_{a_hat} := Delta_min
            self._w_cached = solve_omega_star(gaps, self.closed, a_hat,
                                              lam=self.lam,
                                              solver=self.solver)
            self._n_solves += 1
        self.w_cumsum += self._w_cached

        # averaged D-tracking (their Proposition 3)
        forced = np.where(self.N < np.sqrt(max(self.t, 1)) - self.K / 2.0)[0]
        if forced.size:
            a = int(forced[int(np.argmin(self.N[forced]))])
        else:
            a = int(np.argmin(self.N - self.w_cumsum))
        self._pull(a)
        return None
