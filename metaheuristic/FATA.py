
import numpy as np

def FATA(noP, MaxFEs, lb, ub, dim, fobj):
    """
    FATA: Geophysics-based Optimization Algorithm

    Parameters:
        noP (int): Population size.
        MaxFEs (int): Maximum function evaluations.
        lb (float): Lower bound of variables.
        ub (float): Upper bound of variables.
        dim (int): Dimensionality of the problem.
        fobj (function): Objective function.

    Returns:
        tuple: (Best fitness, Best solution, Convergence curve)
    """
    worstInte = 0    
    bestInte = float('inf')   
    arf = 0.2        
    gBest = np.zeros(dim)
    bestPos = np.zeros(dim)
    cg_curve = []
    gBestScore = float('inf')  
    Flight = np.random.uniform(lb, ub, (noP, dim))  
    fitness = np.full(noP, float('inf'))
    it = 1  
    FEs = 0

    while FEs < MaxFEs:
        for i in range(noP):
            Flight[i, :] = np.clip(Flight[i, :], lb, ub)
            FEs += 1
            fitness[i] = fobj(Flight[i, :])

            if gBestScore > fitness[i]:
                gBestScore = fitness[i]
                gBest = Flight[i, :].copy()

        Index = np.argsort(fitness)
        worstFitness = fitness[Index[-1]]
        bestFitness = fitness[Index[0]]

        Integral = np.cumsum(fitness[Index])
        if Integral[-1] > worstInte:
            worstInte = Integral[-1]
        if Integral[-1] < bestInte:
            bestInte = Integral[-1]
        IP = (Integral[-1] - worstInte) / (bestInte - worstInte + np.finfo(float).eps)

        a = np.tan(-(FEs / MaxFEs) + 1)
        b = 1 / np.tan(-(FEs / MaxFEs) + 1)

        for i in range(noP):
            Para1 = a * np.random.rand(dim) - a * np.random.rand(dim)
            Para2 = b * np.random.rand(dim) - b * np.random.rand(dim)
            p = (fitness[i] - worstFitness) / (gBestScore - worstFitness + np.finfo(float).eps)

            if np.random.rand() > IP:
                Flight[i, :] = np.random.uniform(lb, ub, dim)
            else:
                for j in range(dim):
                    num = np.random.randint(0, noP)
                    if np.random.rand() < p:
                        Flight[i, j] = gBest[j] + Flight[i, j] * Para1[j]
                    else:
                        Flight[i, j] = Flight[num, j] + Para2[j] * Flight[i, j]
                        Flight[i, j] = 0.5 * (arf + 1) * (lb[j] + ub[j]) - arf * Flight[i, j]

        cg_curve.append(gBestScore)
        it += 1
        bestPos = gBest.copy()

    return gBestScore, bestPos, cg_curve
