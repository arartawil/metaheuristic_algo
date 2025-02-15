
import numpy as np

def AFT(noThieves, itemax, lb, ub, dim, fobj):
    ccurve = np.zeros(itemax)
    xth = lb + np.random.rand(noThieves, dim) * (ub - lb)

    fit = np.array([fobj(xth[i, :]) for i in range(noThieves)])
    fitness = fit.copy()
    sorted_indexes = np.argsort(fit)
    sorted_thieves_fitness = fit[sorted_indexes]

    Sorted_thieves = xth[sorted_indexes, :]
    gbest = Sorted_thieves[0, :]
    fit0 = sorted_thieves_fitness[0]

    best = xth.copy()
    xab = xth.copy()

    for ite in range(itemax):
        Pp = 0.1 * np.log(2.75 * (ite / itemax) ** 0.1)
        Td = 2 * np.exp(-2 * (ite / itemax) ** 2)
        
        a = np.ceil((noThieves - 1) * np.random.rand(noThieves)).astype(int)

        for i in range(noThieves):
            if np.random.rand() >= 0.5:
                if np.random.rand() > Pp:
                    xth[i, :] = gbest + (Td * (best[i, :] - xab[i, :]) * np.random.rand() + Td * (xab[i, :] - best[a[i], :]) * np.random.rand()) * np.sign(np.random.rand() - 0.5)
                else:
                    xth[i, :] = Td * ((ub - lb) * np.random.rand(dim) + lb)
            else:
                xth[i, :] = gbest - (Td * (best[i, :] - xab[i, :]) * np.random.rand() + Td * (xab[i, :] - best[a[i], :]) * np.random.rand()) * np.sign(np.random.rand() - 0.5)

        for i in range(noThieves):
            fit[i] = fobj(xth[i, :])
            if np.all(xth[i, :] >= lb) and np.all(xth[i, :] <= ub):
                xab[i, :] = xth[i, :]

                if fit[i] < fitness[i]:
                    best[i, :] = xth[i, :]
                    fitness[i] = fit[i]

                if fitness[i] < fit0:
                    fit0 = fitness[i]
                    gbest = best[i, :]

        ccurve[ite] = fit0

    bestThieves = np.where(fitness == np.min(fitness))[0][0]
    gbestSol = best[bestThieves, :]
    fitness = fobj(gbestSol)

    return fitness, gbestSol, ccurve
