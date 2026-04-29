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


class AlgoBase:
    """
    Spectral bandits [Valko el.at] based graph elimination algorithm with mean estimation using Laplacian.
    """

    def __init__(self, D, A, mu, eta, add_graph=True, eps=0.0):
        """

        Parameters
        ----------
        D : Degree matrix
        A : Adjacency matrix
        mu : node-mean vector
        eta : Penalty parameter for mean estimation
        """

        # TODO : Have not added \epsilon for the \beta confidence width factor. Needed or not?

        self.reset = with_reset
        self.means = mu
        self.eta = eta
        self.D = D
        self.A = A
        self.L = D - A
        self.delta = 0.0001
        self.rho = 0.0001
        self.dim = len(self.L)
        self.eps = eps

        self.remaining_nodes = [i for i in range(self.dim)]
        if add_graph==False:
            self.eta = 0.0
        self.L_rho = self.eta * self.L + self.rho * np.identity(self.dim)
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
        # self.global_tracker_conf_width = []

        self.initialize_conf_width()

    def initialize_conf_width(self):
        """
        Initialize confidence width of all arms.
        """
        v_t_inverse = np.linalg.inv(self.counter + self.L_rho)
        self.inverse_tracker = v_t_inverse
        self.update_conf_width()
        if imperfect_graph_info:
            for i in range(self.dim):
                self.epsilon[i] = support_func.local_eps(self.means, self.L, i)
            self.eps = np.sqrt(self.compute_imperfect_info())
            print("Epsilon error : ", self.eps)

    def compute_imperfect_info(self):
        """
        Computing the error cause of imperfect graph information

        Returns
        -------
        <x, Lx> : quadratic error value

        """

        return support_func.matrix_norm(self.means, self.L)

    def required_reset(self):
        """
        Reset all the arm-counter to 0.
        """
        if self.reset:
            self.counter = np.zeros((self.dim, self.dim))

    def update_conf_width(self):
        """
        Update confidence width of all arms.
        """
        for i in range(self.dim):
            self.conf_width[i] = np.sqrt(self.inverse_tracker[i, i])

    def play_arm(self, index):
        """
        Update counter and reward based on arm played.

        Parameters
        ----------
        index : Arm being played in the current round.

        """
        self.picking_order.append(index)
        counter_vec = np.zeros(self.dim)
        counter_vec[index] = 1
        old_v_t_inverse = self.inverse_tracker
        v_t_inverse = support_func.sherman_morrison_inverse(counter_vec, old_v_t_inverse)
        self.inverse_tracker = v_t_inverse
        self.update_conf_width()

        # FIXME : Testing to remove function "increment_count"
        # #self.increment_count(index)
        self.counter[index, index] += 1
        current_counter = np.array(self.counter)
        # self.counter_tracker.append(current_counter)
        # self.update_conf_width()

        reward = support_func.gaussian_reward(self.means[index])
        self.total_reward[index] = self.total_reward[index] + reward

    def estimate_mean(self):
        """
        Estimate mean using quadratic Laplacian closed form expression.
        """

        self.mean_estimate = np.dot(self.inverse_tracker, self.total_reward)

    def eliminate_arms(self):
        """
        Eliminate arms based on UCB-style argument.
        """

        # TODO : Need to change log(T) to  log(|A_i|)

        beta = 2 * np.sqrt(14 * np.log2(2 * self.dim * np.trace(self.counter) / self.delta)) + 0.5 * self.eta * self.eps
        self.beta_tracker = beta
        temp_array = np.zeros(self.dim)

        # FIXME : Testing commented out code with alternative.
        for i in self.remaining_nodes:
            # beta = 2 * np.sqrt(14 * np.log2(2 * self.dim * np.trace(self.counter) / self.delta)) + 0.5 * self.eta * self.epsilon[i]
            beta = 2 * np.sqrt(14 * np.log2(2 * self.dim * np.trace(self.counter) / self.delta))
            e_i = np.zeros(self.dim)
            e_i[i] = 1.0
            bias = 0.5*self.eta*np.inner(np.dot(self.inverse_tracker, e_i), np.dot(self.L, self.means))
            temp_array[i] = self.mean_estimate[0, i] - beta * self.conf_width[i] +bias

        max_value = max(temp_array)
        self.remaining_nodes = [i for i in self.remaining_nodes if
                                self.mean_estimate[0, i] + beta * self.conf_width[i] >= max_value]

    def play_round(self, num):
        """
        Play the round:
            1. Selecting arm.
            2. Getting reward.
            3. Estimating mean.
            4. Elimination of suboptimal arms.

        Parameters
        ----------
        num : Number of plays before estimation/elimination happens.

        """
        for i in range(num):
            play_index = self.select_arm()
            self.play_arm(play_index)
        # print(len(self.picking_order), len(self.remaining_nodes))

        sys.stdout.flush()
        self.estimate_mean()
        self.eliminate_arms()
        self.required_reset()