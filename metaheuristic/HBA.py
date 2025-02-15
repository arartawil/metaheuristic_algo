import numpy as np

def HBA(N, tmax, lb, ub, dim, objfunc):
    beta = 6
    C = 2
    vec_flag = [1.0, -1.0]

    X = lb + np.random.rand(N, dim) * (ub - lb)
    fitness = np.full(N, float('inf'))

    for i in range(N):
        fitness[i] = objfunc(X[i, :])

    GYbest, gbest = np.min(fitness), np.argmin(fitness)
    Xprey = X[gbest, :]
    Xnew = np.zeros((N, dim))

    CNVG = np.zeros(tmax)

    for t in range(tmax):
        alpha = C * np.exp(-t / tmax)
        I = Intensity(N, Xprey, X)

        for i in range(N):
            r = np.random.rand()
            F = vec_flag[int(np.floor(2 * np.random.rand()))]

            for j in range(dim):
                di = Xprey[j] - X[i, j]

                if r < 0.5:
                    r3, r4, r5 = np.random.rand(), np.random.rand(), np.random.rand()
                    Xnew[i, j] = Xprey[j] + F * beta * I[i] * Xprey[j] + F * r3 * alpha * di * abs(np.cos(2 * np.pi * r4) * (1 - np.cos(2 * np.pi * r5)))
                else:
                    r7 = np.random.rand()
                    Xnew[i, j] = Xprey[j] + F * r7 * alpha * di

            Xnew[i, :] = np.clip(Xnew[i, :], lb, ub)

            tempFitness = objfunc(Xnew[i, :])
            if tempFitness < fitness[i]:
                fitness[i] = tempFitness
                X[i, :] = Xnew[i, :]

        FU = X > ub
        FL = X < lb
        X = (X * ~(FU | FL)) + ub * FU + lb * FL

        Ybest, index = np.min(fitness), np.argmin(fitness)
        CNVG[t] = Ybest
        if Ybest < GYbest:
            GYbest = Ybest
            Xprey = X[index, :]

    return GYbest, Xprey, CNVG

def Intensity(N, Xprey, X):
    di = np.zeros(N)
    S = np.zeros(N)
    I = np.zeros(N)
    eps = 1e-10

    for i in range(N - 1):
        di[i] = np.linalg.norm(X[i, :] - Xprey + eps) ** 2
        S[i] = np.linalg.norm(X[i, :] - X[i + 1, :] + eps) ** 2

    di[N - 1] = np.linalg.norm(X[N - 1, :] - Xprey + eps) ** 2
    S[N - 1] = np.linalg.norm(X[N - 1, :] - X[0, :] + eps) ** 2

    for i in range(N):
        r2 = np.random.rand()
        I[i] = r2 * S[i] / (4 * np.pi * di[i])

    return I