"""
Implementation of the Orthogonal Matching Pursuit Algorithm.
@ntotikos
#"""
import numpy as np
import random
import math

m = 200
N = 175
s = 30

# ground truth
x_true = np.zeros(N)
indices = random.sample(range(N),s) #randomly sample s out of 1000 indices
x_true[indices] = np.random.rand(s)

# random measurement matrix
A = np.random.randn(m,N)
A = A/math.sqrt(m)
# simulated measurements
y = A @ x_true


def get_max_index(residual,matrix,n_clmns):
    matrix_T = matrix.T
    dot_products =[np.dot(matrix_T[i].conjugate(),residual) for i in range(n_clmns)]
    abs_vals = np.absolute(dot_products)
    return np.argmax(abs_vals)


def restrict_on_support(matrix, support_set):
    """Restrict matrix on support set by copying all elements in the columns indicated by
    support_set and setting the other ones to zero."""

    [rows, clms] = matrix.shape
    matrix_supp = np.zeros((rows, clms), dtype='int32')
    matrix_supp[:, support_set] = matrix[:, support_set]
    return matrix_supp


def omp(A, y, iters):
    """Implementation of the Orthogonal Matching Pursuit."""
    from numpy.linalg import inv

    [m, N] = A.shape
    A_H = A.transpose().conjugate()
    # Initializations
    support_set = []
    err_iters = np.zeros(iters)
    x_0 = np.zeros(N)
    r_0 = y

    # Iteration
    for i in range(iters):
        if i == 0:
            residual = r_0
        index = get_max_index(residual, A, N)
        support_set.append(index)

        A_pseudo = inv(A_H @ A) @ A_H
        A_pseudo_r = restrict_on_support(A_pseudo, support_set)
        x_new = A_pseudo_r @ y
        residual = y - A @ x_new

        err = np.linalg.norm(A @ x_new - y, 2)  # l2-error
        err_iters[i] = err
        print(f"err iteration {i}: {err}")

    return x_new, err_iters