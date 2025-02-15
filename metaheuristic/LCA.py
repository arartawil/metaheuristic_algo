import numpy as np
from scipy.special import gamma


def levy(d):
    beta = 1.5
    sigma = (gamma(1 + beta) * np.sin(np.pi * beta / 2) / (gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (
                1 / beta)
    u = np.random.randn(1, d) * sigma
    v = np.random.randn(1, d)
    step = u / np.abs(v) ** (1 / beta)
    return step


def liver_cancer_optimization(n, max_iter, lb, ub, dim, fobj):
    tumor_location = np.zeros(dim)
    tumor_energy = float('inf')
    X = np.random.uniform(lb, ub, (n, dim))
    cnvg = np.zeros(max_iter)

    for t in range(max_iter):
        for i in range(n):
            X[i, :] = np.clip(X[i, :], lb, ub)
            fitness = fobj(X[i, :])

            if fitness < tumor_energy:
                tumor_energy = fitness
                tumor_location = X[i, :].copy()

        r = np.random.rand()
        v = r * t

        for i in range(n):
            f = 1
            l = np.random.rand()
            w = np.random.rand()

            if np.abs(v) <= 5:
                q = np.pi / 6 * (l * w) ** 1.5
                rand_index = np.random.randint(n)
                X_rand = X[rand_index, :]

                if q < 2:
                    X[i, :] = (tumor_location - np.mean(X, axis=0)) - np.random.rand() * (
                                (ub - lb) * np.random.rand() + lb)
                else:
                    X[i, :] = (tumor_location - np.mean(X, axis=0)) - np.random.rand() * (
                                (ub - lb) * np.random.rand() + lb)
            else:
                p = 2 / 3
                jump_strength = v ** p
                X1 = tumor_location - v * np.abs(jump_strength * tumor_location - X[i, :])

                if fobj(X1) < fobj(X[i, :]):
                    X[i, :] = X1
                else:
                    X2 = tumor_location - v * np.abs(jump_strength * tumor_location - X[i, :]) + np.random.rand(
                        dim) * levy(dim)

                    if fobj(X2) < fobj(X[i, :]):
                        X[i, :] = X2

                    X3 = np.random.rand() * X2 + (1 - np.random.rand()) * X1
                    if fobj(X3) < fobj(X[i, :]):
                        X[i, :] = X3

        cnvg[t] = tumor_energy

    return tumor_energy, tumor_location, cnvg
