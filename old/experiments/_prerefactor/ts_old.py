
import numpy as np
import support_func
import algobase_old as algobase
from scipy.stats import norm

class ThompsonSampling(algobase.AlgoBase):
    def __init__(self, D, A, mu, eta, delta, q, eps):
        super().__init__(D, A, mu, eta, eps=eps)
        self.delta = delta
        self.q = q
        self.K = self.dim
        self.converged = False
        self.t = 0
        for arm in range(self.K):
            reward = self.play_arm(arm)
            self.t += 1
    def compute_floor_factor(self, t):
        log_term = np.log(12 * self.K**2 * t**2 / self.delta)
        return np.floor(log_term / self.q)
    def compute_variance_factor(self, t):
        phi_q = norm.isf(self.q)
        return np.log(12 * self.K**2 * t**2 / self.delta) / (phi_q**2)
    def get_all_teff(self):
        M = self.counter + self.L_rho
        M_inv = np.linalg.inv(M)
        return 1.0 / np.diag(M_inv)
    def play_round(self, n_rounds):
        t = np.trace(self.counter)
        self.estimate_mean()
        mu_hat_t = np.asarray(self.mean_estimate).flatten()
        i_hat_t = np.argmax(mu_hat_t)
        teff = self.get_all_teff()
        C_t = self.compute_variance_factor(t=t)
        variances = C_t / teff
        floor = int(self.compute_floor_factor(t=t))
        i_tilde_t_m = np.zeros(floor)
        delta_hat_t_m = np.zeros((floor, self.K))
        for m in range(floor):
            theta_m_current = np.random.normal(mu_hat_t, np.sqrt(variances))
            i_tilde_t_m[m] = np.argmax(theta_m_current)
            delta_hat_t_m[m] = theta_m_current - mu_hat_t
        if np.all(i_tilde_t_m == i_hat_t):
            self.converged = True
            self.remaining_nodes = [i_hat_t]
            return i_hat_t
        max_per_row = np.max(delta_hat_t_m, axis=1)
        m_star = np.argmax(max_per_row)
        i_tilde_t = int(i_tilde_t_m[m_star])
        if self.counter[i_tilde_t, i_tilde_t] < self.counter[i_hat_t, i_hat_t]:
            self.play_arm(i_tilde_t)
        else:
            self.play_arm(i_hat_t)
