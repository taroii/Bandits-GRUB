"""
PSD Kernel Algorithm Library

Implements Kernel-TS-Explore (Algorithm 3) from Chang 2026.
Replaces the combinatorial Laplacian L_G with a general PSD kernel K_G
(default: normalized Laplacian K_G = I - D^{-1/2} A D^{-1/2}).
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import graph_algo


class KernelThompsonSampling(graph_algo.ThompsonSampling):
    """
    Kernel-TS-Explore: identical to TS-Explore except the regularizer
    uses K_G instead of L_G in the effective sample size computation.
    """

    def __init__(self, D, A, mu, eta, delta, q, eps, kernel=None):
        """
        Parameters
        ----------
        D : Degree matrix
        A : Adjacency matrix
        mu : node-mean vector
        eta : Penalty parameter for mean estimation
        delta : error constraint
        q : threshold parameter in [delta, 0.1]
        eps : smoothness bound
        kernel : PSD kernel matrix K_G. If None, uses normalized Laplacian.
        """
        if kernel is not None:
            self.K_G = np.asarray(kernel, dtype=float)
        else:
            # Normalized Laplacian: K_G = I - D^{-1/2} A D^{-1/2}
            D_arr = np.asarray(D, dtype=float)
            A_arr = np.asarray(A, dtype=float)
            n = len(mu)
            deg = np.array([D_arr[i, i] for i in range(n)])
            # Handle isolated nodes (degree 0) by treating 1/sqrt(0) as 0
            with np.errstate(divide='ignore', invalid='ignore'):
                D_inv_sqrt = np.diag(np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0))
            self.K_G = np.eye(n) - D_inv_sqrt @ A_arr @ D_inv_sqrt

        # Initialize base class (sets self.L = D - A, etc.)
        super().__init__(D, A, mu, eta, delta, q, eps)

    def get_teff(self, i):
        """
        t_eff,i = 1 / [(V_t^{-1})_{ii}]
        where V_t = Σ e_πt e_πt^T + ρ K_G  (using K_G instead of L_G)
        """
        M = self.counter + self.rho * np.asmatrix(self.K_G)
        M_inv = np.linalg.inv(M)
        return 1.0 / M_inv[i, i]
