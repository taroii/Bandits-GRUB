"""
Misspecified Graph Algorithm Library

Implements TS-Explore under graph misspecification (Section 4.4) from Chang 2026.
Same algorithm as base TS-Explore, but the Thompson sampling variance is inflated
by an additive bias term (rho * eps_mis)^2 / t_eff_i^2 per arm.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import graph_algo
import support_func


class MisspecifiedThompsonSampling(graph_algo.ThompsonSampling):
    """
    TS-Explore under misspecified graph smoothness.

    The variance for each arm's Thompson sample is inflated from
        C(t) / t_eff_i
    to
        C(t) / t_eff_i  +  (rho * eps_mis)^2 / t_eff_i^2
    to absorb the additive bias in the confidence radius.
    """

    def __init__(self, D, A, mu, eta, delta, q, eps_mis=None):
        """
        Parameters
        ----------
        D : Degree matrix
        A : Adjacency matrix
        mu : node-mean vector
        eta : Penalty parameter for mean estimation
        delta : error constraint
        q : threshold parameter in [delta, 0.1]
        eps_mis : Misspecification level. If None, defaults to half
                  the true smoothness sqrt(mu^T L mu).
        """
        # Compute true smoothness before super().__init__ resets eps
        L = np.asarray(D) - np.asarray(A)
        true_eps = np.sqrt(support_func.matrix_norm(mu, L))

        if eps_mis is None:
            self.eps_mis = 0.001 * true_eps
        else:
            self.eps_mis = eps_mis

        print(f"True smoothness: {true_eps:.4f}, eps_mis: {self.eps_mis:.4f}")

        super().__init__(D, A, mu, eta, delta, q, eps=0.0)

    def get_thompson_variance(self, t, i):
        """
        Inflated variance for arm i's Thompson sample:
            C(t) / t_eff_i  +  (rho * eps_mis)^2 / t_eff_i^2
        """
        C_t = self.compute_variance_factor(t)
        t_eff_i = self.get_teff(i)
        bias_term = (self.rho * self.eps_mis) ** 2 / (t_eff_i ** 2)
        return C_t / t_eff_i + bias_term

    def play_round(self, n_rounds):
        t = np.trace(self.counter)

        self.estimate_mean()
        mu_hat_t = np.asarray(self.mean_estimate).flatten()
        i_hat_t = np.argmax(mu_hat_t)

        floor = int(self.compute_floor_factor(t=t))
        i_tilde_t_m = np.zeros(floor)
        delta_hat_t_m = np.zeros((floor, self.K))
        for m in range(floor):
            theta_m_current = np.zeros(self.K)
            for i in range(self.K):
                variance = self.get_thompson_variance(t, i)
                theta_m_current[i] = np.random.normal(
                    mu_hat_t[i],
                    variance**0.5
                )
            i_tilde_t_m[m] = np.argmax(theta_m_current)
            delta_hat_t_m[m] = theta_m_current - mu_hat_t

        if np.all(i_tilde_t_m == i_hat_t):
            self.converged = True
            self.remaining_nodes = [i_hat_t]
            return i_hat_t
        else:
            max_per_row = np.max(delta_hat_t_m, axis=1)
            m_star = np.argmax(max_per_row)
            i_tilde_t = int(i_tilde_t_m[m_star])

            if self.counter[i_tilde_t, i_tilde_t] < self.counter[i_hat_t, i_hat_t]:
                self.play_arm(i_tilde_t)
            else:
                self.play_arm(i_hat_t)
