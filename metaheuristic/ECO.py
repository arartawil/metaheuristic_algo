
import numpy as np

def ECO(N, Max_iter, lb, ub, dim, fobj):
    """
    Educational Competition Optimizer (ECO)

    Parameters:
        N (int): Population size.
        Max_iter (int): Maximum iterations.
        lb (float): Lower bound of variables.
        ub (float): Upper bound of variables.
        dim (int): Dimensionality of the problem.
        fobj (function): Objective function.

    Returns:
        tuple: (Best fitness, Best solution, Convergence curve)
    """
    G1Number = int(0.2 * N)
    G2Number = int(0.1 * N)

    X = np.random.uniform(lb, ub, (N, dim))
    fitness = np.array([fobj(X[i, :]) for i in range(N)])
    index = np.argsort(fitness)
    X = X[index, :]
    fitness = fitness[index]

    GBestF = fitness[0]
    GBestX = X[0, :]

    Convergence_curve = np.zeros(Max_iter)

    for i in range(Max_iter):
        R1 = np.random.rand()
        R2 = np.random.rand()
        P = 4 * np.random.randn() * (1 - i / Max_iter)
        E = (np.pi * i) / (P * Max_iter)
        w = 0.1 * np.log(2 - (i / Max_iter))

        for j in range(N):
            if i % 3 == 1:
                if j < G1Number:
                    X[j, :] += w * (np.mean(X[j, :]) - X[j, :]) * np.random.standard_cauchy(dim)
                else:
                    X[j, :] += w * (GBestX - X[j, :]) * np.random.randn(dim)
            
            elif i % 3 == 2:
                if j < G2Number:
                    X[j, :] += (GBestX - np.mean(X, axis=0)) * np.exp(i / Max_iter - 1) * np.random.standard_cauchy(dim)
                else:
                    if R1 < 0.5:
                        X[j, :] -= w * GBestX - P * (E * w * GBestX - X[j, :])
                    else:
                        X[j, :] -= w * GBestX - P * (w * GBestX - X[j, :])

            else:
                if j < G2Number:
                    X[j, :] += (GBestX - X[j, :]) * np.random.randn(dim) - (GBestX - X[j, :]) * np.random.randn(dim)
                else:
                    if R2 < 0.5:
                        X[j, :] = GBestX - P * (E * GBestX - X[j, :])
                    else:
                        X[j, :] = GBestX - P * (GBestX - X[j, :])

            X[j, :] = np.clip(X[j, :], lb, ub)
            new_fitness = fobj(X[j, :])

            if new_fitness < fitness[j]:
                fitness[j] = new_fitness

            if new_fitness < GBestF:
                GBestF = new_fitness
                GBestX = X[j, :]

        Convergence_curve[i] = GBestF

    return GBestF, GBestX, Convergence_curve
