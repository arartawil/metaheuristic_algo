
import numpy as np
from scipy.stats import norm

def levyF(Popsize, Dim, beta):
    sigma_u = (np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2) / 
               (np.math.gamma((1 + beta) / 2) * beta * 2**((beta - 1) / 2)))**(1 / beta)
    u = np.random.randn(Popsize, Dim) * sigma_u
    v = np.random.randn(Popsize, Dim)
    step = u / np.abs(v)**(1 / beta)
    return step

def DO(Popsize, Maxiteration, LB, UB, Dim, Fobj):
    """
    Dandelion Optimizer (DO)

    Parameters:
        Popsize (int): Population size.
        Maxiteration (int): Maximum iterations.
        LB (float): Lower bound of variables.
        UB (float): Upper bound of variables.
        Dim (int): Dimensionality of the problem.
        Fobj (function): Objective function.

    Returns:
        tuple: (Best fitness, Best solution, Convergence curve)
    """
    dandelions = np.random.uniform(LB, UB, (Popsize, Dim))
    dandelionsFitness = np.array([Fobj(dandelions[i, :]) for i in range(Popsize)])
    Convergence_curve = np.zeros(Maxiteration)

    sorted_indexes = np.argsort(dandelionsFitness)
    Best_position = dandelions[sorted_indexes[0], :]
    Best_fitness = dandelionsFitness[sorted_indexes[0]]
    Convergence_curve[0] = Best_fitness

    for t in range(1, Maxiteration):
        beta = np.random.randn(Popsize, Dim)
        alpha = np.random.rand() * ((1 / Maxiteration**2) * t**2 - 2 / Maxiteration * t + 1)
        a, b, c = 1 / (Maxiteration**2 - 2 * Maxiteration + 1), -2 / Maxiteration, 1 - (1 / Maxiteration**2 - 2 / Maxiteration + 1)
        k = 1 - np.random.rand() * (c + a * t**2 + b * t)

        if np.random.randn() < 1.5:
            dandelions_1 = dandelions.copy()
            for i in range(Popsize):
                lamb = np.abs(np.random.randn(Dim))
                theta = (2 * np.random.rand() - 1) * np.pi
                row = 1 / np.exp(theta)
                vx = row * np.cos(theta)
                vy = row * np.sin(theta)
                NEW = np.random.uniform(LB, UB, Dim)
                dandelions_1[i, :] += alpha * vx * vy * norm.pdf(lamb) * (NEW - dandelions[i, :])
        else:
            dandelions_1 = dandelions * k

        dandelions = np.clip(dandelions_1, LB, UB)

        dandelions_mean = np.mean(dandelions, axis=0)
        dandelions_2 = dandelions.copy()
        for i in range(Popsize):
            dandelions_2[i, :] -= beta[i, :] * alpha * (dandelions_mean - beta[i, :] * alpha * dandelions[i, :])

        dandelions = np.clip(dandelions_2, LB, UB)

        Step_length = levyF(Popsize, Dim, 1.5)
        Elite = np.tile(Best_position, (Popsize, 1))
        dandelions_3 = dandelions.copy()
        for i in range(Popsize):
            dandelions_3[i, :] = Elite[i, :] + Step_length[i, :] * alpha * (Elite[i, :] - dandelions[i, :] * (2 * t / Maxiteration))

        dandelions = np.clip(dandelions_3, LB, UB)

        dandelionsFitness = np.array([Fobj(dandelions[i, :]) for i in range(Popsize)])
        sorted_indexes = np.argsort(dandelionsFitness)
        dandelions = dandelions[sorted_indexes, :]
        SortfitbestN = dandelionsFitness[sorted_indexes]

        if SortfitbestN[0] < Best_fitness:
            Best_position = dandelions[0, :]
            Best_fitness = SortfitbestN[0]

        Convergence_curve[t] = Best_fitness

    return Best_fitness, Best_position, Convergence_curve
