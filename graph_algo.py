"""
Graph Algorithm Library

Algorithms used for the GraphBandits simulations.
"""

import numpy as np
import support_func
import algobase
from scipy.stats import norm

with_reset = False
imperfect_graph_info = True


# ---------------------------------------------------------------------------
# UCB-style baselines
# ---------------------------------------------------------------------------

class MaxVarianceArmAlgo(algobase.AlgoBase):
    """Spectral-bandits arm sampling from the remaining set."""

    def select_arm(self):
        remaining_width = np.zeros(self.dim)
        for i in self.remaining_nodes:
            remaining_width[i] = self.conf_width[i]
        return int(np.argmax(remaining_width))


class NoGraphAlgo(algobase.AlgoBase):
    """Cyclic/UCB baseline without graph regularization."""

    def __init__(self, D, A, mu, rho_lap=0.0, **kwargs):
        kwargs.pop('add_graph', None)
        super().__init__(D, A, mu, rho_lap=rho_lap, add_graph=False, **kwargs)

    def select_arm(self):
        remaining_width = np.zeros(self.dim)
        for i in self.remaining_nodes:
            remaining_width[i] = self.conf_width[i]
        return int(np.argmax(remaining_width))


class CyclicAlgo(algobase.AlgoBase):
    """Cyclic arm selection from the remaining set (jumping across clusters)."""

    def select_arm(self):
        next_index = 0
        play_index = self.remaining_nodes[next_index % len(self.remaining_nodes)]

        if len(self.picking_order) > 1:
            last_index = self.picking_order[-1]
            ind = np.where(self.jumping_index == last_index)
            ind = (int(ind[0]) + 1) % len(self.jumping_index)
            while self.jumping_index[ind] not in self.remaining_nodes:
                ind += 1
                ind %= len(self.jumping_index)
            play_index = self.jumping_index[ind]

        self.picking_order.append(play_index)
        return play_index


class MaxDiffVarAlgo(algobase.AlgoBase):
    """Ensemble confidence-width reduction selection (MVM / JVM-O paper variant)."""

    def opti_selection(self):
        A = self.remaining_nodes
        options = []
        for i in A:
            new_vec = np.zeros(self.dim)
            new_vec[i] = 1
            current = support_func.sherman_morrison_inverse(new_vec, self.inverse_tracker)
            options.append(np.linalg.det(current))
        index = int(np.argmin(options))
        return np.array(A)[index]

    def select_arm(self):
        return self.opti_selection()


class OneStepMinDetAlgo(algobase.AlgoBase):
    def opti_selection(self):
        A = self.remaining_nodes
        options = []
        for i in A:
            new_vec = np.zeros(self.dim)
            new_vec[i] = 1
            current = support_func.sherman_morrison_inverse(new_vec, self.inverse_tracker)
            options.append(np.linalg.det(current))
        index = int(np.argmin(options))
        return np.array(A)[index]

    def select_arm(self):
        return self.opti_selection()


class OneStepMinSumAlgo(algobase.AlgoBase):
    def opti_selection(self):
        A = self.remaining_nodes
        options = []
        for i in A:
            new_vec = np.zeros(self.dim)
            new_vec[i] = 1
            current = support_func.sherman_morrison_inverse(new_vec, self.inverse_tracker)
            options.append(sum([current[j, j] for j in A]))
        index = int(np.argmin(options))
        return np.array(A)[index]

    def select_arm(self):
        return self.opti_selection()


# ---------------------------------------------------------------------------
# Thompson sampling variants
# ---------------------------------------------------------------------------

class BasicThompsonSampling:
    """No-graph Thompson sampling baseline (empirical means, direct counts)."""

    def __init__(self, mu, delta, q):
        self.means = np.asarray(mu, dtype=float).flatten()
        self.K = len(self.means)
        self.delta = delta
        self.q = q
        self.converged = False
        self.t = 0

        self.counts = np.zeros(self.K)
        self.total_reward = np.zeros(self.K)
        self.remaining_nodes = list(range(self.K))
        self.pull_counts = np.zeros(self.K)

        for arm in range(self.K):
            reward = support_func.gaussian_reward(self.means[arm])
            self.counts[arm] += 1
            self.total_reward[arm] += reward
            self.t += 1

    def compute_floor_factor(self, t):
        log_term = np.log(12 * self.K**2 * max(t, 1)**2 / self.delta)
        return np.floor(max(log_term, 1.0) / self.q)

    def compute_variance_factor(self, t):
        phi_q = norm.isf(self.q)
        return np.log(12 * self.K**2 * max(t, 1)**2 / self.delta) / (phi_q**2)

    def play_round(self, n_rounds=1):
        t = self.t
        mu_hat = self.total_reward / np.maximum(self.counts, 1)
        i_hat = int(np.argmax(mu_hat))

        C_t = self.compute_variance_factor(t)
        variances = C_t / np.maximum(self.counts, 1)

        floor = max(int(self.compute_floor_factor(t)), 1)
        i_tilde_m = np.zeros(floor, dtype=int)
        delta_hat_m = np.zeros((floor, self.K))
        for m in range(floor):
            theta = np.random.normal(mu_hat, np.sqrt(variances))
            i_tilde_m[m] = int(np.argmax(theta))
            delta_hat_m[m] = theta - mu_hat

        if np.all(i_tilde_m == i_hat):
            self.converged = True
            self.remaining_nodes = [i_hat]
            return i_hat

        m_star = int(np.argmax(np.max(delta_hat_m, axis=1)))
        i_tilde = int(i_tilde_m[m_star])
        arm = i_tilde if self.counts[i_tilde] < self.counts[i_hat] else i_hat

        reward = support_func.gaussian_reward(self.means[arm])
        self.counts[arm] += 1
        self.total_reward[arm] += reward
        self.t += 1
        self.pull_counts[arm] += 1
        return None


class ThompsonSampling(algobase.AlgoBase):
    """
    Graph-smooth Thompson Sampling (Chang et al. 2026).

    Parameters
    ----------
    D, A, mu          Problem definition.
    rho_lap           Paper's rho (Laplacian weight).
    delta, q          Confidence level and TS-sample quantile.
    epsilon_nominal   Nominal smoothness used in the confidence radius.
                      If None, auto-computes <mu, L mu>^{1/2} from the true means.
    kernel            'combinatorial' or 'normalized'.
    rho_diag          Diagonal regularizer (scale up when rho_lap is huge).
    """

    def __init__(self, D, A, mu, rho_lap, delta, q, epsilon_nominal=None,
                 kernel='combinatorial', rho_diag=1e-4):
        super().__init__(D, A, mu, rho_lap=rho_lap,
                         epsilon_nominal=epsilon_nominal,
                         kernel=kernel, rho_diag=rho_diag)
        self.delta = delta
        self.q = q
        self.K = self.dim
        self.converged = False
        self.t = 0

        for arm in range(self.K):
            self.play_arm(arm)
            self.t += 1

    def compute_floor_factor(self, t):
        t_safe = max(float(t), 1.0)
        log_term = np.log(12 * self.K**2 * t_safe**2 / self.delta)
        return np.floor(max(log_term, 1.0) / self.q)

    def compute_variance_factor(self, t):
        t_safe = max(float(t), 1.0)
        phi_q = norm.isf(self.q)
        return np.log(12 * self.K**2 * t_safe**2 / self.delta) / (phi_q**2)

    def get_all_teff(self):
        diag = np.diag(self.inverse_tracker)
        return 1.0 / np.maximum(diag, 1e-300)

    def get_R(self):
        return self.total_reward

    def play_round(self, n_rounds=1):
        t = np.trace(self.counter)
        self.estimate_mean()
        mu_hat_t = np.asarray(self.mean_estimate).flatten()
        i_hat_t = int(np.argmax(mu_hat_t))

        teff = self.get_all_teff()
        C_t = self.compute_variance_factor(t=t)
        variances = C_t / teff

        floor = max(int(self.compute_floor_factor(t=t)), 1)
        i_tilde_t_m = np.zeros(floor, dtype=int)
        delta_hat_t_m = np.zeros((floor, self.K))
        for m in range(floor):
            theta_m = np.random.normal(mu_hat_t, np.sqrt(np.maximum(variances, 0.0)))
            i_tilde_t_m[m] = int(np.argmax(theta_m))
            delta_hat_t_m[m] = theta_m - mu_hat_t

        if np.all(i_tilde_t_m == i_hat_t):
            self.converged = True
            self.remaining_nodes = [i_hat_t]
            return i_hat_t

        m_star = int(np.argmax(np.max(delta_hat_t_m, axis=1)))
        i_tilde_t = int(i_tilde_t_m[m_star])

        if self.counter[i_tilde_t, i_tilde_t] < self.counter[i_hat_t, i_hat_t]:
            self.play_arm(i_tilde_t)
        else:
            self.play_arm(i_hat_t)
        self.t += 1
        return None


# ---------------------------------------------------------------------------
# Graph-feedback Thompson sampling (Algorithm 2 in §4.3)
# ---------------------------------------------------------------------------

def _greedy_dominating_set(Adj):
    """Greedy dominating set on the closed neighborhoods of A+I."""
    A = np.asarray(Adj, dtype=float)
    n = A.shape[0]
    closed = (A + np.eye(n)) > 0
    covered = np.zeros(n, dtype=bool)
    dom = []
    while not covered.all():
        new_cover = np.array(
            [(closed[a] & ~covered).sum() for a in range(n)]
        )
        a = int(np.argmax(new_cover))
        dom.append(a)
        covered |= closed[a]
    return dom


class GraphFeedbackTS:
    """
    Thompson sampling for the graph-feedback setting (§4.3).

    Each pull of arm ``a`` reveals an i.i.d. reward for every arm in
    ``N+(a) = {a} ∪ {j : A[a, j] > 0}``.  The estimator is the empirical
    mean of the feedback observations.
    """

    def __init__(self, D, A, mu, delta, q, sigma=1.0):
        self.means = np.asarray(mu, dtype=float).flatten()
        self.Adj = np.asarray(A, dtype=float)
        self.K = len(self.means)
        self.delta = delta
        self.q = q
        self.sigma = sigma
        self.converged = False
        self.t = 0

        self.closed = (self.Adj + np.eye(self.K)) > 0
        self.N_fb = np.zeros(self.K)
        self.R_fb = np.zeros(self.K)
        self.pull_counts = np.zeros(self.K)
        self.remaining_nodes = list(range(self.K))

        for a in _greedy_dominating_set(self.Adj):
            self._pull(a)

    def _pull(self, a):
        a = int(a)
        nbrs = np.where(self.closed[a])[0]
        for j in nbrs:
            r = self.means[j] + self.sigma * np.random.randn()
            self.N_fb[j] += 1
            self.R_fb[j] += r
        self.pull_counts[a] += 1
        self.t += 1

    def compute_floor_factor(self, t):
        t_safe = max(float(t), 1.0)
        log_term = np.log(12 * self.K**2 * t_safe**2 / self.delta)
        return np.floor(max(log_term, 1.0) / self.q)

    def compute_variance_factor(self, t):
        t_safe = max(float(t), 1.0)
        phi_q = norm.isf(self.q)
        return (self.sigma**2) * np.log(12 * self.K**2 * t_safe**2 / self.delta) / (phi_q**2)

    def play_round(self, n_rounds=1):
        t = self.t
        N_safe = np.maximum(self.N_fb, 1)
        mu_hat = self.R_fb / N_safe
        i_hat = int(np.argmax(mu_hat))

        C_t = self.compute_variance_factor(t)
        variances = C_t / N_safe

        floor = max(int(self.compute_floor_factor(t)), 1)
        i_tilde_m = np.zeros(floor, dtype=int)
        delta_hat_m = np.zeros((floor, self.K))
        for m in range(floor):
            theta = np.random.normal(mu_hat, np.sqrt(np.maximum(variances, 0.0)))
            i_tilde_m[m] = int(np.argmax(theta))
            delta_hat_m[m] = theta - mu_hat

        if np.all(i_tilde_m == i_hat):
            self.converged = True
            self.remaining_nodes = [i_hat]
            return i_hat

        m_star = int(np.argmax(np.max(delta_hat_m, axis=1)))
        i_tilde = int(i_tilde_m[m_star])
        candidates = {i_hat, i_tilde}

        # Pick action a that covers the most candidates, ties broken by smaller N_fb.
        best_a, best_key = -1, None
        for a in range(self.K):
            cover = int(self.closed[a, list(candidates)].sum())
            key = (-cover, self.N_fb[a])
            if best_key is None or key < best_key:
                best_key = key
                best_a = a
        self._pull(best_a)
        return None
