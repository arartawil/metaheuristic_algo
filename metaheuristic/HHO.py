import numpy as np
from scipy.special import gamma


def levy(d):
    beta = 1.5
    sigma = (gamma(1 + beta) * np.sin(np.pi * beta / 2) / (gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (
                1 / beta)
    u = np.random.randn(d) * sigma
    v = np.random.randn(d)
    step = u / np.abs(v) ** (1 / beta)
    return step


def hho(N, T, lb, ub, dim, fobj):
    Rabbit_Location = np.zeros(dim)
    Rabbit_Energy = float('inf')

    X = np.random.uniform(lb, ub, (N, dim))
    CNVG = np.zeros(T)

    for t in range(T):
        for i in range(N):
            X[i, :] = np.clip(X[i, :], lb, ub)
            fitness = fobj(X[i, :])

            if fitness < Rabbit_Energy:
                Rabbit_Energy = fitness
                Rabbit_Location = np.copy(X[i, :])

        E1 = 2 * (1 - (t / T))

        for i in range(N):
            E0 = 2 * np.random.rand() - 1
            Escaping_Energy = E1 * E0

            if np.abs(Escaping_Energy) >= 1:
                q = np.random.rand()
                rand_Hawk_index = np.random.randint(0, N)
                X_rand = X[rand_Hawk_index, :]

                if q < 0.5:
                    X[i, :] = X_rand - np.random.rand() * np.abs(X_rand - 2 * np.random.rand() * X[i, :])
                else:
                    X[i, :] = (Rabbit_Location - np.mean(X, axis=0)) - np.random.rand() * (
                                (ub - lb) * np.random.rand() + lb)
            else:
                r = np.random.rand()

                if r >= 0.5 and np.abs(Escaping_Energy) < 0.5:
                    X[i, :] = Rabbit_Location - Escaping_Energy * np.abs(Rabbit_Location - X[i, :])
                elif r >= 0.5 and np.abs(Escaping_Energy) >= 0.5:
                    Jump_strength = 2 * (1 - np.random.rand())
                    X[i, :] = (Rabbit_Location - X[i, :]) - Escaping_Energy * np.abs(
                        Jump_strength * Rabbit_Location - X[i, :])
                elif r < 0.5 and np.abs(Escaping_Energy) >= 0.5:
                    Jump_strength = 2 * (1 - np.random.rand())
                    X1 = Rabbit_Location - Escaping_Energy * np.abs(Jump_strength * Rabbit_Location - X[i, :])

                    if fobj(X1) < fobj(X[i, :]):
                        X[i, :] = X1
                    else:
                        X2 = Rabbit_Location - Escaping_Energy * np.abs(
                            Jump_strength * Rabbit_Location - X[i, :]) + np.random.rand(dim) * levy(dim)
                        if fobj(X2) < fobj(X[i, :]):
                            X[i, :] = X2
                elif r < 0.5 and np.abs(Escaping_Energy) < 0.5:
                    Jump_strength = 2 * (1 - np.random.rand())
                    X1 = Rabbit_Location - Escaping_Energy * np.abs(
                        Jump_strength * Rabbit_Location - np.mean(X, axis=0))

                    if fobj(X1) < fobj(X[i, :]):
                        X[i, :] = X1
                    else:
                        X2 = Rabbit_Location - Escaping_Energy * np.abs(
                            Jump_strength * Rabbit_Location - np.mean(X, axis=0)) + np.random.rand(dim) * levy(dim)
                        if fobj(X2) < fobj(X[i, :]):
                            X[i, :] = X2

        CNVG[t] = Rabbit_Energy

    return Rabbit_Energy, Rabbit_Location, CNVG
