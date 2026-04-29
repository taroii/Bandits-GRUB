"""
Supporting Function Library

Contains function definitions required for a more generic use-case.
"""


import numpy as np
import math
from scipy.optimize import minimize


def jumping_list(clusters, tot):
    num_cluster = len(clusters)
    # print(clusters)
    # print(sum([len(clu) for clu in clusters]))
    pick_order = []
    i=-1
    while len(pick_order) < tot:
        i = i+1
        for j in range(num_cluster):
            try:
                a = clusters[j][i]
                # print(tot, a, len(pick_order))
                pick_order.append(a)
            except IndexError:
                pass
    return pick_order


def get_clusters(Adj):
    dim = len(Adj)
    label = np.array([i for i in range(dim)])
    for j in range(dim):
        for i in range(dim):
            if Adj[i, j] != 0:
                if i >= j:
                    label[i] = label[j]
                else:
                    label[j] = label[i]

    clusters_label = set(label)
    clusters = []
    for i in clusters_label:
        new_cluster = np.where(label == i)[0]
        clusters.append(new_cluster)

    return clusters


def sherman_morrison_inverse(x, V):
    vec = np.dot(x, V)
    constant = 1 + matrix_norm(x, V)
    # print(len(V), constant[0,0], len(np.outer(vec,vec)))
    # exit()
    try:
        return_mat = V - (1./constant[0, 0])*np.outer(vec, vec)
    except IndexError:
        return_mat = V - (1./constant)*np.outer(vec, vec)

    return return_mat

def local_eps(x, V, i):
    """
    Compute quadratic function value <x, Vx>.

    Parameters
    ----------
    x : vector
    V : Matrix

    Returns
    -------
    float : quadratic function value <x, Vx>
    """
    total = 0
    for j in range(len(V)):
        if V[i, j]!=0:
            total += (x[i]-x[j])**2
    return math.sqrt(total)


def matrix_norm(x, V):
    """
    Compute quadratic function value <x, Vx>.

    Parameters
    ----------
    x : vector
    V : Matrix

    Returns
    -------
    float : quadratic function value <x, Vx>
    """

    return np.dot(x, np.dot(V, x).T)


def gaussian_reward(mu_i, mag=1.0):
    """
    Generate gaussian rewards given mean and variance. This is only for 1-dim case.

    Parameters
    ----------
    mu_i : mean value
    mag : variance value

    Returns
    -------
    float : gaussian reward with given mean and variance
    """

    return mu_i + np.random.randn()*mag


def round_function(i):
    """
    Number of arms to be played in a round before Estimation/Elimination routine.

    Parameters
    ----------
    i : round number

    Returns
    -------
    int : number of arms to be played in the current round
    """

    # return 2*i
    # return 2**i
    return 1


def laplacian_error(x, L, eps):
    # print(abs(matrix_norm(x, L) - eps), np.linalg.norm(matrix_norm(x, L) - eps))
    return np.linalg.norm(matrix_norm(x, L) - eps)


def laplacian_jac(x, L, eps):
    # print(int(np.sign(matrix_norm(x, L) - eps)))
    # return int(np.sign(matrix_norm(x, L) - eps))*L
    return L

def find_means(L, eps, x0):
    res = minimize(laplacian_error, x0, method='Nelder-Mead', args=(L, eps), tol=0.001, options={'disp' : True})
    mean_vector = res.x
    print(res.message)
    return mean_vector


def find_means_2(L, eps, x0):
    res = minimize(laplacian_error, x0, method='BFGS', jac=laplacian_jac, args=(L, eps), tol=0.1, options={'disp' : True})
    mean_vector = res.x
    print(res.message)
    return mean_vector