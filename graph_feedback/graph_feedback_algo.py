"""
Graph Feedback Algorithm Library

Implements TS-Explore-GF (Algorithm 2) from Chang 2026.
When arm a is pulled, rewards of all arms in N_G^+(a) are observed.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
from scipy.stats import norm
import support_func


class GraphFeedbackTS:
    """
    Thompson Sampling for graph feedback (TS-Explore-GF).

    Pulling arm a reveals rewards of all arms in its closed neighborhood N_G^+(a).
    The estimator is the empirical mean R_i^fb / N_i^fb.
    Arm selection picks the action whose neighborhood best covers {i_hat, i_tilde}.
    """

    def __init__(self, D, A, mu, delta, q):
        """
        Parameters
        ----------
        D : Degree matrix
        A : Adjacency matrix (np.matrix or np.array)
        mu : node-mean vector
        delta : error constraint
        q : threshold parameter in [delta, 0.1]
        """
        self.means = mu
        self.A = np.asarray(A)
        self.D = np.asarray(D)
        self.K = len(mu)
        self.delta = delta
        self.q = q
        self.converged = False
        self.t = 0

        # Build closed neighborhood lists: N_G^+(a) = neighbors(a) ∪ {a}
        self.neighborhoods = []
        for a in range(self.K):
            neighbors = [a]
            for j in range(self.K):
                if j != a and self.A[a, j] > 0:
                    neighbors.append(j)
            self.neighborhoods.append(neighbors)

        # Feedback observation counts and cumulative rewards
        self.N_fb = np.zeros(self.K)
        self.R_fb = np.zeros(self.K)
        self.direct_pulls = np.zeros(self.K)

        # For tracking in run_algo
        self.remaining_nodes = list(range(self.K))

        # Initialize: pull a covering set so every arm is observed at least once
        self._initial_cover()

    def _initial_cover(self):
        """Pull a set of arms whose closed neighborhoods cover [K]."""
        covered = set()
        uncovered = set(range(self.K))
        while uncovered:
            # Greedy: pick arm that covers the most uncovered arms
            best_arm = max(range(self.K),
                           key=lambda a: len(uncovered & set(self.neighborhoods[a])))
            self._pull_and_observe(best_arm)
            covered.update(self.neighborhoods[best_arm])
            uncovered -= set(self.neighborhoods[best_arm])

    def _pull_and_observe(self, arm):
        """Pull arm and observe rewards for all arms in N_G^+(arm)."""
        self.t += 1
        self.direct_pulls[arm] += 1
        for j in self.neighborhoods[arm]:
            reward = support_func.gaussian_reward(self.means[j])
            self.R_fb[j] += reward
            self.N_fb[j] += 1

    def compute_M(self, t):
        """M(delta, q, t) = floor(1/q * log(12 K^2 t^2 / delta))"""
        log_term = np.log(12 * self.K**2 * t**2 / self.delta)
        return max(1, int(np.floor(log_term / self.q)))

    def compute_C(self, t):
        """C(delta, q, t) = log(12 K^2 t^2 / delta) / phi^2(q)"""
        phi_q = norm.isf(self.q)
        return np.log(12 * self.K**2 * t**2 / self.delta) / (phi_q**2)

    def play_round(self, n_rounds):
        """Execute one round of TS-Explore-GF."""
        t = self.t

        # Compute empirical means
        mu_hat = np.zeros(self.K)
        for i in range(self.K):
            if self.N_fb[i] > 0:
                mu_hat[i] = self.R_fb[i] / self.N_fb[i]

        i_hat = np.argmax(mu_hat)

        # Draw M Thompson samples
        M = self.compute_M(t)
        C_t = self.compute_C(t)

        i_tilde_samples = np.zeros(M, dtype=int)
        delta_hat = np.zeros(M)

        for m in range(M):
            theta = np.zeros(self.K)
            for i in range(self.K):
                variance = C_t / max(self.N_fb[i], 1)
                theta[i] = np.random.normal(mu_hat[i], np.sqrt(variance))

            i_tilde_samples[m] = np.argmax(theta)
            delta_hat[m] = theta[i_tilde_samples[m]] - theta[i_hat]

        # Stopping condition: all sampled maximizers agree with empirical best
        if np.all(i_tilde_samples == i_hat):
            self.converged = True
            self.remaining_nodes = [i_hat]
            return i_hat

        # Find challenger with largest disagreement margin
        m_star = np.argmax(delta_hat)
        i_tilde = i_tilde_samples[m_star]

        # Choose action whose N_G^+(a) covers {i_hat, i_tilde}, break ties by min obs count
        candidates = {i_hat, i_tilde}
        best_action = None
        best_cover = -1
        best_min_obs = float('inf')

        for a in range(self.K):
            cover = len(candidates & set(self.neighborhoods[a]))
            if cover > best_cover or (cover == best_cover and self.direct_pulls[a] < best_min_obs):
                best_cover = cover
                best_action = a
                best_min_obs = self.direct_pulls[a]

        self._pull_and_observe(best_action)
