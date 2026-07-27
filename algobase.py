"""
Graph Algorithm Library

This file contains all the algorithms used for the GraphBandits simulations.
All algorithms can be used as modules.

"""

import sys
import numpy as np
import support_func

with_reset = False
imperfect_graph_info = True


def build_kernel(D, A, kernel='combinatorial'):
    """
    Build the graph-kernel matrix used for Laplacian regularization.

    kernel='combinatorial' : L_G = D - A
    kernel='normalized'    : K_G = I - D^{-1/2} A D^{-1/2}   (isolated nodes dropped)
    """
    D = np.asarray(D, dtype=float)
    A = np.asarray(A, dtype=float)
    dim = D.shape[0]
    if kernel == 'combinatorial':
        return D - A
    if kernel == 'normalized':
        d = np.diag(D)
        d_inv_sqrt = np.where(d > 0, 1.0 / np.sqrt(np.maximum(d, 1e-12)), 0.0)
        Dis = np.diag(d_inv_sqrt)
        return np.eye(dim) - Dis @ A @ Dis
    raise ValueError(f"unknown kernel={kernel!r}")


class AlgoBase:
    """
    Spectral bandits [Valko el.at] based graph elimination algorithm
    with mean estimation using a Laplacian regularizer.

    Parameters
    ----------
    D, A              Degree and adjacency matrices (shape (K, K))
    mu                True node-mean vector, length K
    rho_lap           Laplacian weight (paper's rho). Set to 0 if add_graph=False.
    add_graph         If False, drops the graph regularization entirely.
    epsilon_nominal   Nominal smoothness used in the confidence radius. If None,
                      it is auto-computed from the true means via <mu, L mu>^{1/2}.
    kernel            'combinatorial' (D-A) or 'normalized' (I - D^{-1/2} A D^{-1/2}).
    rho_diag          Small diagonal regularizer to keep V_t invertible in the
                      null direction of L_G. Defaults to 1e-4. Increase when
                      rho_lap is very large (Experiment 3 auto-scales this).
    grub_bias         Which bias convention the UCB--LCB elimination radius
                      uses (see ``eliminate_arms``).  One of:

                      'legacy'    -- reproduces the committed paper numbers:
                                     log2 noise term, and half the *signed*
                                     exact oracle bias added to the lower
                                     bound only.  DEFAULT, so existing
                                     results are bit-for-bit reproducible.
                      'published' -- beta_i = 2 sigma sqrt(14 log(2Kt/delta))
                                     + rho*eps, applied symmetrically.  This
                                     is the radius of Thaker et al. (2022),
                                     Lemma 2 / Alg. 1, and the radius our
                                     Appendix H.1 claims to use.
                      'sqrt'      -- same but with sqrt(rho)*eps, the tight
                                     bias implied by V_t >= rho L_G (our own
                                     Section 3.2 argument).
                      'oracle'    -- noise term plus the exact bias magnitude
                                     |[V^-1 rho L_G mu]_i|, symmetric.
                                     Oracle-strength; an upper bound on how
                                     well any bias handling can do.
                      'none'      -- noise term only (no bias correction).
    grub_log          Base of the logarithm in the noise term: 'ln' (the
                      published expression), 'log2' (what the legacy code
                      computes, which inflates the radius by 1.201x), or
                      'auto' = log2 for grub_bias='legacy', ln otherwise.
    """

    _GRUB_BIAS_CHOICES = ('legacy', 'published', 'sqrt', 'oracle', 'none')

    def __init__(self, D, A, mu, rho_lap, add_graph=True,
                 epsilon_nominal=None, kernel='combinatorial',
                 rho_diag=1e-4, delta=1e-4, reward_fn=None,
                 grub_bias='legacy', grub_log='auto'):

        if grub_bias not in self._GRUB_BIAS_CHOICES:
            raise ValueError(f"grub_bias={grub_bias!r} not in "
                             f"{self._GRUB_BIAS_CHOICES}")
        if grub_log not in ('auto', 'ln', 'log2'):
            raise ValueError(f"grub_log={grub_log!r}")
        self.grub_bias = grub_bias
        self.grub_log = ('log2' if grub_bias == 'legacy' else 'ln') \
            if grub_log == 'auto' else grub_log

        self.reset = with_reset
        self.means = np.asarray(mu, dtype=float).flatten()
        self.D = D
        self.A = A
        self.rho_lap = float(rho_lap) if add_graph else 0.0
        self.rho_diag = float(rho_diag)
        self.delta = float(delta)
        self.kernel = kernel
        # Optional environment-specific reward sampler:
        # callable(arm_idx) -> scalar reward.  Defaults to N(mu_arm, sigma=1).
        self.reward_fn = reward_fn
        self.L = build_kernel(D, A, kernel=kernel)
        self.dim = self.L.shape[0]
        self.epsilon_nominal = epsilon_nominal
        self.eps = 0.0 if epsilon_nominal is None else float(epsilon_nominal)

        self.remaining_nodes = [i for i in range(self.dim)]
        self.L_rho = self.rho_lap * self.L + self.rho_diag * np.identity(self.dim)
        self.counter = np.zeros((self.dim, self.dim))
        self.conf_width = np.zeros(self.dim)
        self.total_reward = np.zeros(self.dim)
        self.mean_estimate = np.zeros(self.dim)
        self.clusters = support_func.get_clusters(self.A)
        self.jumping_index = np.array(support_func.jumping_list(self.clusters, self.dim))
        self.epsilon = np.zeros(self.dim)

        self.beta_tracker = 0.0
        self.inverse_tracker = np.zeros((self.dim, self.dim))
        self.picking_order = []
        # Task 1 instrumentation hooks (read by experiments.utils.tracing).
        # UCB-style elimination has no disagreement pair, hence (-1, -1).
        self.last_pair = (-1, -1)
        self.last_pull = -1

        # Precompute (0.5 * rho * L * mu); constant for the whole run, used in
        # eliminate_arms.  Avoids rebuilding L @ mu every step.
        self._half_rho_L_mu = 0.5 * self.rho_lap * np.dot(self.L, self.means)

        self.initialize_conf_width()

    # Backwards-compat alias: older code referred to `self.eta`.
    @property
    def eta(self):
        return self.rho_lap

    def initialize_conf_width(self):
        v_t_inverse = np.linalg.inv(self.counter + self.L_rho)
        self.inverse_tracker = v_t_inverse
        self.update_conf_width()
        if imperfect_graph_info:
            for i in range(self.dim):
                self.epsilon[i] = support_func.local_eps(self.means, self.L, i)
            if self.epsilon_nominal is None:
                computed = self.compute_imperfect_info()
                self.eps = float(np.sqrt(np.maximum(computed, 0.0)))

    def compute_imperfect_info(self):
        """Quadratic Laplacian error <mu, L mu>."""
        val = support_func.matrix_norm(self.means, self.L)
        return float(np.asarray(val).flatten()[0])

    def required_reset(self):
        if self.reset:
            self.counter = np.zeros((self.dim, self.dim))

    def update_conf_width(self):
        self.conf_width = np.sqrt(np.maximum(np.diag(self.inverse_tracker), 0.0))

    def _draw_reward(self, arm):
        """Sample one reward for ``arm``.  Override via ``reward_fn`` to use
        a non-Gaussian environment (e.g., empirical user ratings).
        """
        if self.reward_fn is None:
            return support_func.gaussian_reward(self.means[arm])
        return float(self.reward_fn(arm))

    def play_arm(self, index):
        self.picking_order.append(index)
        counter_vec = np.zeros(self.dim)
        counter_vec[index] = 1
        self.inverse_tracker = support_func.sherman_morrison_inverse(
            counter_vec, self.inverse_tracker)
        self.update_conf_width()
        self.counter[index, index] += 1
        reward = self._draw_reward(index)
        self.total_reward[index] += reward

    def estimate_mean(self):
        self.mean_estimate = np.dot(self.inverse_tracker, self.total_reward)

    def _noise_multiplier(self, t_count):
        """GRUB noise term 2*sigma*sqrt(14*log(2Kt/delta)), sigma = 1.

        The published expression (Thaker et al. 2022) uses the natural log;
        ``grub_log='log2'`` reproduces the legacy implementation, which
        inflates the multiplier by sqrt(1/ln 2) = 1.201.
        """
        log_arg = max(2.0 * self.dim * t_count / self.delta, 2.0)
        lg = np.log2(log_arg) if self.grub_log == 'log2' else np.log(log_arg)
        return 2.0 * np.sqrt(14.0 * max(lg, 0.0))

    def eliminate_arms(self):
        t_count = np.trace(self.counter)
        conc_radius = self._noise_multiplier(t_count)
        mu_hat = np.asarray(self.mean_estimate).flatten()

        if self.grub_bias == 'legacy':
            # Historical behaviour, kept as the default so the committed
            # figures reproduce exactly.  Note this is NOT the radius that
            # Appendix H.1 describes: it uses half the *signed* exact oracle
            # bias 0.5*rho*<V^-1 e_i, L mu> on the lower bound only, rather
            # than the published symmetric rho*eps/sqrt(t_eff,i).
            self.beta_tracker = conc_radius + 0.5 * self.rho_lap * self.eps
            bias_vec = np.dot(self.inverse_tracker, self._half_rho_L_mu)
            upper = mu_hat + conc_radius * self.conf_width
            lower = mu_hat - conc_radius * self.conf_width + bias_vec
        else:
            # Symmetric radius: |mu_hat_i - mu_i| <= (noise + bias_i)
            #                                        * sqrt([V^-1]_ii).
            # conf_width already holds sqrt([V^-1]_ii) = 1/sqrt(t_eff,i).
            if self.grub_bias == 'published':
                bias_r = self.rho_lap * self.eps * self.conf_width
            elif self.grub_bias == 'sqrt':
                bias_r = np.sqrt(self.rho_lap) * self.eps * self.conf_width
            elif self.grub_bias == 'oracle':
                bias_r = np.abs(np.dot(self.inverse_tracker,
                                       2.0 * self._half_rho_L_mu))
            else:  # 'none'
                bias_r = np.zeros(self.dim)
            self.beta_tracker = conc_radius + self.rho_lap * self.eps
            radius = conc_radius * self.conf_width + bias_r
            upper = mu_hat + radius
            lower = mu_hat - radius

        rem = np.fromiter(self.remaining_nodes, dtype=int, count=len(self.remaining_nodes))
        max_value = float(np.max(lower[rem]))
        self.remaining_nodes = [i for i in self.remaining_nodes if upper[i] >= max_value]

    def play_round(self, num):
        for _ in range(num):
            play_index = self.select_arm()
            self.last_pull = int(play_index)
            self.play_arm(play_index)
        sys.stdout.flush()
        self.estimate_mean()
        self.eliminate_arms()
        self.required_reset()
