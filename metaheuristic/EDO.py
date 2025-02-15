
import numpy as np

def initialization(N, Dim, UB, LB):
    return np.random.uniform(LB, UB, (N, Dim))

def EDO(N, Max_iter, LB, UB, Dim, F_obj):
    """
    Exponential Distribution Optimizer (EDO)

    Parameters:
        N (int): Population size.
        Max_iter (int): Maximum iterations.
        LB (float): Lower bound of variables.
        UB (float): Upper bound of variables.
        Dim (int): Dimensionality of the problem.
        F_obj (function): Objective function.

    Returns:
        tuple: (Best fitness, Best solution, Convergence curve)
    """
    BestSol = np.zeros(Dim)
    BestFitness = float('inf')
    Xwinners = initialization(N, Dim, UB, LB)
    Fitness = np.array([F_obj(Xwinners[i, :]) for i in range(N)])

    for i in range(N):
        if Fitness[i] < BestFitness:
            BestFitness = Fitness[i]
            BestSol = Xwinners[i, :].copy()

    Memoryless = Xwinners.copy()
    iter = 0
    cgcurve = np.zeros(Max_iter)

    while iter < Max_iter:
        V = np.zeros((N, Dim))
        cgcurve[iter] = BestFitness

        sorted_indices = np.argsort(Fitness)
        Fitness = Fitness[sorted_indices]
        Xwinners = Xwinners[sorted_indices, :]

        d = (1 - iter / Max_iter)
        f = 2 * np.random.rand() - 1
        a = f ** 10
        b = f ** 5
        c = d * f
        X_guide = np.mean(Xwinners[:3, :], axis=0)

        for i in range(N):
            alpha = np.random.rand()
            if alpha < 0.5:
                if np.all(Memoryless[i, :] == Xwinners[i, :]):
                    Mu = (X_guide + Memoryless[i, :]) / 2.0
                    ExP_rate = 1.0 / Mu
                    variance = 1.0 / (ExP_rate ** 2)
                    V[i, :] = a * (Memoryless[i, :] - variance) + b * X_guide
                else:
                    Mu = (X_guide + Memoryless[i, :]) / 2.0
                    ExP_rate = 1.0 / Mu
                    variance = 1.0 / (ExP_rate ** 2)
                    phi = np.random.rand()
                    V[i, :] = b * (Memoryless[i, :] - variance) + np.log(phi) * Xwinners[i, :]
            else:
                M = np.mean(Xwinners, axis=0)
                s = np.random.permutation(N)
                D1 = M - Xwinners[s[0], :]
                D2 = M - Xwinners[s[1], :]
                Z1 = M - D1 + D2
                Z2 = M - D2 + D1
                V[i, :] = Xwinners[i, :] + c * Z1 + (1.0 - c) * Z2 - M

            V[i, :] = np.clip(V[i, :], LB, UB)

        Memoryless = V.copy()

        V_Fitness = np.array([F_obj(V[i, :]) for i in range(N)])
        for i in range(N):
            if V_Fitness[i] < Fitness[i]:
                Xwinners[i, :] = V[i, :]
                Fitness[i] = V_Fitness[i]
                if Fitness[i] < BestFitness:
                    BestFitness = Fitness[i]
                    BestSol = Xwinners[i, :].copy()

        iter += 1

    return BestFitness, BestSol, cgcurve
