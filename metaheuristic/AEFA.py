import numpy as np
from .utils import clamp


def AEFA(N, max_it, lb, ub, D, benchmark, tag=1):
    """
    Artificial Electric Field Algorithm (AEFA) for Global Optimization.

    Parameters:
        N (int): Number of agents.
        max_it (int): Maximum iterations.
        lb (float or np.array): Lower bounds.
        ub (float or np.array): Upper bounds.
        D (int): Dimension of the problem.
        benchmark (function): Fitness evaluation function.
        tag (int): Optimization direction (1 for minimization, -1 for maximization).

    Returns:
        tuple: (Best fitness value, Best solution, List of best values over iterations)
    """

    Rnorm = 2
    FCheck = 1
    Rpower = 1

    Fbest = np.inf if tag == 1 else -np.inf  # Initialize best fitness
    Lbest = np.zeros(D)  # Initialize best solution

    # Initialize search agents
    X = np.random.rand(N, D) * (ub - lb) + lb
    V = np.zeros((N, D))
    fitness = np.zeros(N)

    BestValues = []
    MeanValues = []

    for iteration in range(max_it):
        # Evaluate fitness
        for i in range(N):
            fitness[i] = benchmark(X[i, :])

        # Find best solution
        if tag == 1:
            best_X = np.argmin(fitness)
        else:
            best_X = np.argmax(fitness)

        best = fitness[best_X]

        # Update best solution
        if iteration == 0 or (tag == 1 and best < Fbest) or (tag == -1 and best > Fbest):
            Fbest = best
            Lbest = X[best_X, :].copy()

        BestValues.append(Fbest)
        MeanValues.append(np.mean(fitness))

        # Calculate charge Q
        Fmax, Fmin, Fmean = np.max(fitness), np.min(fitness), np.mean(fitness)

        if Fmax == Fmin:
            Q = np.ones(N)
        else:
            best, worst = (Fmin, Fmax) if tag == 1 else (Fmax, Fmin)
            Q = np.exp((fitness - worst) / (best - worst))

        Q /= np.sum(Q)

        # Define the best fraction
        fper = 3
        cbest = round(N * (fper + (1 - iteration / max_it) * (100 - fper)) / 100) if FCheck == 1 else N
        sorted_indices = np.argsort(Q)[::-1]  # Sort descending
        Qs = Q[sorted_indices]

        # Compute electric force
        E = np.zeros((N, D))
        for i in range(N):
            for ii in range(cbest):
                j = sorted_indices[ii]
                if j != i:
                    R = np.linalg.norm(X[i, :] - X[j, :], ord=Rnorm)  # Distance
                    for k in range(D):
                        E[i, k] += np.random.rand() * Q[j] * ((X[j, k] - X[i, k]) / (R ** Rpower + np.finfo(float).eps))

        # Compute acceleration
        alfa = 30
        K0 = 500
        K = K0 * np.exp(-alfa * iteration / max_it)
        a = E * K

        # Update velocity and position
        V = np.random.rand(N, D) * V + a
        X = X + V
        X = clamp(X, lb, ub)  # Apply boundary constraints

    return Fbest, Lbest, BestValues
