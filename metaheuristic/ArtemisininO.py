
import numpy as np

def Mutation(z, x, b, dim):
    for j in range(dim):
        if np.random.rand() < 0.05:
            z[j] = x[j]
        if np.random.rand() < 0.2:
            z[j] = b[j]
    return z

def Transborder_reset(z, ub, lb, dim, best):
    for j in range(dim):
        if z[j] > ub or z[j] < lb:
            z[j] = best[j]
    return z

def ArtemisininO(N, MaxIter, lb, ub, dim, fobj):
    FEs = 0
    MaxFEs = N * MaxIter
    pop = np.random.rand(N, dim) * (ub - lb) + lb
    Fitness = np.array([fobj(pop[i, :]) for i in range(N)])
    FEs += N
    fmin = np.min(Fitness)
    best = pop[np.argmin(Fitness), :]

    Convergence_curve = []

    while FEs < MaxFEs:
        K = 1 - ((FEs**(1/6)) / (MaxFEs**(1/6)))
        E = np.exp(-4 * (FEs / MaxFEs))

        New_pop = pop.copy()

        for i in range(N):
            Fitnorm = (Fitness[i] - np.min(Fitness)) / (np.max(Fitness) - np.min(Fitness))

            for j in range(dim):
                if np.random.rand() < K:
                    if np.random.rand() < 0.5:
                        New_pop[i, j] = pop[i, j] + E * pop[i, j] * (-1) ** FEs
                    else:
                        New_pop[i, j] = pop[i, j] + E * best[j] * (-1) ** FEs

                if np.random.rand() < Fitnorm:
                    A = np.random.permutation(N)
                    beta = (np.random.rand() / 2) + 0.1
                    New_pop[i, j] = pop[A[2], j] + beta * (pop[A[0], j] - pop[A[1], j])

            New_pop[i, :] = Mutation(New_pop[i, :], pop[i, :], best, dim)
            New_pop[i, :] = Transborder_reset(New_pop[i, :], ub, lb, dim, best)

            tFitness = fobj(New_pop[i, :])
            FEs += 1

            if tFitness < Fitness[i]:
                pop[i, :] = New_pop[i, :]
                Fitness[i] = tFitness

        fmin = np.min(Fitness)
        best = pop[np.argmin(Fitness), :]

        Convergence_curve.append(fmin)

    return fmin, best, Convergence_curve
