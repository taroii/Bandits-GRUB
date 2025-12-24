"""
Graph Algorithm Library

This file contains all the algorithms used for the GraphBandits simulations.
All algorithms can be used as modules.

"""

import numpy as np
import support_func
import algobase

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
    Thompson Sampling algorithm for graph bandits based on TS-Explore from the paper.
    """

    def __init__(self, D, A, mu, eta, delta=0.0001, q=0.01, eps=0.0):
        """
        Parameters
        ----------
        D : Degree matrix
        A : Adjacency matrix
        mu : node-mean vector
        eta : Penalty parameter for mean estimation
        delta : Confidence parameter
        q : Threshold parameter for sampling
        eps : Epsilon for imperfect graph info
        """
        super().__init__(D, A, mu, eta, eps=eps)
        self.q = q
        self.M = 1  # Number of posterior samples
        self.converged = False  # Track convergence status
        
    def compute_variance_factor(self, t):
        """
        Compute C(δ, q, t) from the paper - variance scaling factor.
        """
        if t <= 0:
            t = 1
        # C(δ, q, t) = log(12K²t²/δ) / φ²(q)
        # Using approximation: φ²(q) ≈ 2*log(1/q) for small q
        phi_squared = 2 * np.log(1.0 / self.q) if self.q > 0 else 1.0
        C_t = np.log(12 * self.dim**2 * t**2 / self.delta) / phi_squared
        return max(C_t, 1.0)  # Ensure positive variance
        
    def compute_num_samples(self, t):
        """
        Compute M = floor(1/q * log(12K²t²/δ)) - number of posterior samples.
        """
        if t <= 0:
            return 1
        M = int(np.floor((1.0 / self.q) * np.log(12 * self.dim**2 * t**2 / self.delta)))
        return max(1, min(M, 100))  # Cap at 100 for computational efficiency

    def opti_selection(self):
        """
        Thompson Sampling selection using posterior sampling.
        Implements exchange set mechanism from TS-Explore algorithm.
        """
        # Get total number of samples
        t = int(np.trace(self.counter)) + 1
        
        # Compute number of posterior samples
        self.M = self.compute_num_samples(t)
        
        # Get empirical best arm
        # Handle both 1D and 2D mean_estimate arrays
        if len(self.mean_estimate.shape) == 2:
            mean_values = self.mean_estimate[0]
        else:
            mean_values = self.mean_estimate
            
        empirical_best = np.argmax(mean_values)
        if empirical_best not in self.remaining_nodes:
            # If empirical best was eliminated, pick from remaining
            remaining_means = np.array([mean_values[i] if i in self.remaining_nodes else -np.inf 
                                       for i in range(self.dim)])
            empirical_best = np.argmax(remaining_means)
        
        # Compute variance scaling factor
        C_t = self.compute_variance_factor(t)
        
        # Track the sample with maximum disagreement
        max_gap = -np.inf
        selected_challenger = empirical_best
        all_samples_agree = True
        
        # Cache mean_estimate shape check for performance
        if len(self.mean_estimate.shape) == 2:
            mean_values_cached = self.mean_estimate[0]
        else:
            mean_values_cached = self.mean_estimate
        
        for sample_idx in range(self.M):
            # Sample from posterior for each arm
            theta_sample = np.zeros(self.dim)
            
            for i in self.remaining_nodes:
                # Use effective number of plays (approximated by direct plays + regularization)
                effective_plays = self.counter[i,i] + self.rho
                
                # Variance for posterior sampling
                variance = C_t / max(1.0, effective_plays)
                
                # Sample from Gaussian posterior (using cached mean values)
                theta_sample[i] = np.random.normal(
                    mean_values_cached[i], 
                    np.sqrt(variance)
                )
            
            # Set eliminated arms to -inf
            for i in range(self.dim):
                if i not in self.remaining_nodes:
                    theta_sample[i] = -np.inf
            
            # Find sampled best arm
            sampled_best = np.argmax(theta_sample)
            
            # Check for disagreement
            if sampled_best != empirical_best:
                all_samples_agree = False
                # Compute gap for this sample
                gap = theta_sample[sampled_best] - theta_sample[empirical_best]
                if gap > max_gap:
                    max_gap = gap
                    selected_challenger = sampled_best
                # Early break if we found disagreement and don't need to track all samples
                if self.M > 10:  # Only break early for large M
                    break
        
        # Update convergence status
        self.converged = all_samples_agree
        
        # Select least sampled arm from exchange set {empirical_best, selected_challenger}
        if selected_challenger == empirical_best:
            return empirical_best
        elif self.counter[empirical_best, empirical_best] <= self.counter[selected_challenger, selected_challenger]:
            return empirical_best
        else:
            return selected_challenger

    def eliminate_arms(self):
        """
        Override elimination for Thompson Sampling - no elimination based on UCB bounds.
        Thompson Sampling maintains all arms until consensus is reached.
        """
        pass  # No elimination for Thompson Sampling
    
    def has_converged(self):
        """
        Check if Thompson Sampling has converged (all samples agree on best arm).
        """
        return self.converged
    
    def select_arm(self):
        """
        Select arm using Thompson Sampling with exchange set mechanism.
        """
        # Ensure we have valid remaining nodes
        if len(self.remaining_nodes) == 0:
            return 0
        elif len(self.remaining_nodes) == 1:
            return self.remaining_nodes[0]
        
        return self.opti_selection()