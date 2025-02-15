
import numpy as np

def DBO(pop, M, c, d, dim, fobj):
    """
    Dung Beetle Optimizer (DBO) Algorithm

    Parameters:
        pop (int): Population size.
        M (int): Maximum iterations.
        c (float): Lower bound of variables.
        d (float): Upper bound of variables.
        dim (int): Dimensionality of the problem.
        fobj (function): Objective function.

    Returns:
        tuple: (Best fitness, Best solution, Convergence curve)
    """
    P_percent = 0.2
    pNum = round(pop * P_percent)
    
    lb = np.full(dim, c)
    ub = np.full(dim, d)
    
    x = [lb + (ub - lb) * np.random.rand(dim) for _ in range(pop)]
    fit = np.array([fobj(x[i]) for i in range(pop)])
    pFit = fit.copy()
    pX = x.copy()
    XX = pX.copy()
    
    bestI = np.argmin(fit)
    fMin = fit[bestI]
    bestX = x[bestI].copy()
    Convergence_curve = np.zeros(M)

    for t in range(M):
        B = np.argmax(fit)
        worse = x[B]
        r2 = np.random.rand()

        for i in range(int(pNum)):
            if r2 < 0.9:
                r1 = np.random.rand()
                a = 1 if np.random.rand() > 0.1 else -1
                x[i] = pX[i] + 0.3 * np.abs(pX[i] - worse) + a * 0.1 * XX[i]
            else:
                theta = np.random.choice([0, 90, 180]) * np.pi / 180
                x[i] = pX[i] + np.tan(theta) * np.abs(pX[i] - XX[i])
            x[i] = np.clip(x[i], lb, ub)
            fit[i] = fobj(x[i])

        bestII = np.argmin(fit)
        fMMin = fit[bestII]
        bestXX = x[bestII].copy()
        R = 1 - t / M

        Xnew1 = np.clip(bestXX * (1 - R), lb, ub)
        Xnew2 = np.clip(bestXX * (1 + R), lb, ub)
        Xnew11 = np.clip(bestX * (1 - R), lb, ub)
        Xnew22 = np.clip(bestX * (1 + R), lb, ub)

        for i in range(int(pNum), min(12, pop)):
            x[i] = bestXX + np.random.rand(dim) * (pX[i] - Xnew1) + np.random.rand(dim) * (pX[i] - Xnew2)
            x[i] = np.clip(x[i], Xnew1, Xnew2)
            fit[i] = fobj(x[i])

        for i in range(13, min(19, pop)):
            x[i] = pX[i] + np.random.randn() * (pX[i] - Xnew11) + np.random.rand(dim) * (pX[i] - Xnew22)
            x[i] = np.clip(x[i], lb, ub)
            fit[i] = fobj(x[i])

        for j in range(20, pop):
            x[j] = bestX + np.random.randn(dim) * ((np.abs(pX[j] - bestXX) + np.abs(pX[j] - bestX)) / 2)
            x[j] = np.clip(x[j], lb, ub)
            fit[j] = fobj(x[j])

        XX = pX.copy()
        for i in range(pop):
            if fit[i] < pFit[i]:
                pFit[i] = fit[i]
                pX[i] = x[i].copy()
            if pFit[i] < fMin:
                fMin = pFit[i]
                bestX = pX[i].copy()

        Convergence_curve[t] = fMin

    return fMin, bestX, Convergence_curve
