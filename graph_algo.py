"""
Graph Algorithm Library

This file contains all the algorithms used for the GraphBandits simulations.
All algorithms can be used as modules.

"""

import numpy as np
import support_func
import algobase
from scipy.stats import norm

with_reset = False
imperfect_graph_info = True


class MaxVarianceArmAlgo(algobase.AlgoBase):
    """
    Cyclic algorithm with mean estimation using Laplacian.
    """
    def select_arm(self):
        """
        Spectral bandits [Valko el.at] based arm sampling from the remaining set of arms.
        """
        remaining_width = np.zeros(self.dim)
        for i in self.remaining_nodes:
            remaining_width[i] = self.conf_width[i]
        play_index = np.argmax(remaining_width)

        return play_index


class NoGraphAlgo(algobase.AlgoBase):
    """
    Cyclic algorithm with mean estimation using Laplacian.
    """

    def __init__(self, D, A, mu, eta):
        """

        Parameters
        ----------
        D : Degree matrix
        A : Adjacency matrix
        mu : node-mean vector
        eta : Penalty parameter for mean estimation

        """
        super().__init__(D, A, mu, eta, add_graph=False)

    def select_arm(self):
        """
        Cyclic arm selection from the remaining set of arms.
        """

        # TODO : Switch current algorithm with true cyclic. Currently only works for symmetric graphs.

        remaining_width = np.zeros(self.dim)
        for i in self.remaining_nodes:
            remaining_width[i] = self.conf_width[i]
        play_index = np.argmax(remaining_width)

        return play_index


class CyclicAlgo(algobase.AlgoBase):
    """
    Spectral bandits [Valko el.at] based graph elimination algorithm with mean estimation using Laplacian.
    """

    def select_arm(self):
        """
        Spectral bandits [Valko el.at] based arm sampling from the remaining set of arms.
        """

        next_index= 0
        play_index = self.remaining_nodes[next_index%len(self.remaining_nodes)]

        if len(self.picking_order) > 1:
            last_index = self.picking_order[-1]
            ind = np.where(self.jumping_index == last_index)
            ind = (int(ind[0]) + 1)%len(self.jumping_index)
            while self.jumping_index[ind] not in self.remaining_nodes:
                ind +=1
                ind = ind%len(self.jumping_index)
            play_index = self.jumping_index[ind]

        self.picking_order.append(play_index)

        return play_index


class MaxDiffVarAlgo(algobase.AlgoBase):
    """
    Proposed graph elimination algorithm with mean estimation using Laplacian.
    """

    def opti_selection(self):
        """
        Proposed arm selection criteria based on the ensemble reduction of confidence width.
        """

        # TODO : Replace costly inverse computation using Sherman-Morrison formula.
        A = self.remaining_nodes
        options =[]
        for i in A:
            new_vec = np.zeros(self.dim)
            new_vec[i] = 1
            current = support_func.sherman_morrison_inverse(new_vec, self.inverse_tracker)
            options.append(np.linalg.det(current))
        index = np.argmin(options)
        return np.array(A)[index]

    def select_arm(self):
        """
        Select arm to play based on proposed ensemble confidence width reduction criteria.
        """
        remaining_width = np.zeros(self.dim)
        for i in self.remaining_nodes:
            remaining_width[i] = self.conf_width[i]
        play_index = self.opti_selection()

        return play_index


class OneStepMinDetAlgo(algobase.AlgoBase):
    """
    Proposed graph elimination algorithm with mean estimation using Laplacian.
    """

    def opti_selection(self):
        """
        Proposed arm selection criteria based on the ensemble reduction of confidence width.
        """

        # TODO : Replace costly inverse computation using Sherman-Morrison formula.
        A = self.remaining_nodes
        options =[]
        for i in A:
            new_vec = np.zeros(self.dim)
            new_vec[i] = 1
            current = support_func.sherman_morrison_inverse(new_vec, self.inverse_tracker)
            options.append(np.linalg.det(current))
        index = np.argmin(options)
        return np.array(A)[index]

    def select_arm(self):
        """
        Select arm to play based on proposed ensemble confidence width reduction criteria.
        """
        remaining_width = np.zeros(self.dim)
        for i in self.remaining_nodes:
            remaining_width[i] = self.conf_width[i]
        play_index = self.opti_selection()

        return play_index


class OneStepMinSumAlgo(algobase.AlgoBase):
    """
    Proposed graph elimination algorithm with mean estimation using Laplacian.
    """

    def opti_selection(self):
        """
        Proposed arm selection criteria based on the ensemble reduction of confidence width.
        """

        # TODO : Replace costly inverse computation using Sherman-Morrison formula.
        A = self.remaining_nodes
        options =[]
        for i in A:
            new_vec = np.zeros(self.dim)
            new_vec[i] = 1
            current = support_func.sherman_morrison_inverse(new_vec, self.inverse_tracker)
            options.append(sum([current[j, j] for j in A]))
        index = np.argmin(options)
        return np.array(A)[index]

    def select_arm(self):
        """
        Select arm to play based on proposed ensemble confidence width reduction criteria.
        """
        remaining_width = np.zeros(self.dim)
        for i in self.remaining_nodes:
            remaining_width[i] = self.conf_width[i]
        play_index = self.opti_selection()

        return play_index


class ThompsonSampling(algobase.AlgoBase):
    """
    Thompsoon sampling algorithm for graph bandits based on Chang et al. 2026.
    """

    def __init__(self, D, A, mu, eta, delta, q, eps):
        super().__init__(D, A, mu, eta, eps=eps)
        self.delta = delta
        self.q = q
        self.K = self.dim
        self.converged = False
        self.t = 0

        # play through first round
        for arm in range(self.K):
            reward = self.play_arm(arm)
            self.t += 1
    
    def compute_floor_factor(self, t):
        """
        floor( 1/q * log(12 K^2 t^2 / delta) )
        """
        log_term = np.log(12 * self.K**2 * t**2 / self.delta)
        return np.floor(log_term / self.q)
    
    def compute_variance_factor(self, t):
        """
        C(delta, q, t) = log(12 K^2 t^2 / delta) / phi^2(q)
        """
        phi_q = norm.isf(self.q)
        C_t = np.log(12 * self.K**2 * t**2 / self.delta) / (phi_q**2)
        return C_t
    
    def get_teff(self):
        return np.diag(self.counter)
    
    def get_R(self):
        return self.total_reward

    def play_round(self, n_rounds):
        #! TODO
        # t counter is incremented outside of class
        t_eff = np.diag(self.counter)
        t = np.sum(t_eff)
        
        # Calculate \hat{i}_t
        self.estimate_mean()
        # Convert from numpy.matrix to numpy.array and flatten
        mu_hat_t = np.asarray(self.mean_estimate).flatten()
        i_hat_t = np.argmax(mu_hat_t)

        # For loops
        floor = int(self.compute_floor_factor(t=t))
        i_tilde_t_m = np.zeros(floor)
        delta_hat_t_m = np.zeros((floor, self.K))
        for m in range(floor):
            # Sample from posterior for each arm
            theta_m_current = np.zeros(self.K)
            for i in range(self.K):
                variance = self.compute_variance_factor(t=t) / t_eff[i]
                theta_m_current[i] = np.random.normal(
                    mu_hat_t[i],
                    variance**0.5
                )
            i_tilde_t_m[m] =  np.argmax(theta_m_current)
            delta_hat_t_m[m] = theta_m_current - mu_hat_t

        # Conditional statements
        if np.all(i_tilde_t_m == i_hat_t):
            self.converged = True
            self.remaining_nodes = [i_hat_t]
            return i_hat_t
        else:
            # Find which sample (m) had the largest difference for any arm
            max_per_row = np.max(delta_hat_t_m, axis=1)  # Max difference in each row
            m_star = np.argmax(max_per_row)  # Row with largest max difference
            i_tilde_t = int(i_tilde_t_m[m_star])
            self.play_arm(i_tilde_t)