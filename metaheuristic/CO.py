
import numpy as np

def CO(n, MaxIt, lb, ub, dim, fobj):
    BestCost = np.inf
    BestSol = np.zeros(dim)

    pop = np.random.rand(n, dim) * (ub - lb) + lb
    popCost = np.array([fobj(pop[i, :]) for i in range(n)])

    for i in range(n):
        if popCost[i] < BestCost:
            BestCost = popCost[i]
            BestSol = pop[i, :].copy()

    BestCostArray = np.zeros(MaxIt)

    for t in range(MaxIt):
        for i in range(n):
            Xnew = pop[i, :] + 0.1 * np.random.randn(dim) * (BestSol - pop[i, :])
            Xnew = np.clip(Xnew, lb, ub)
            newCost = fobj(Xnew)

            if newCost < popCost[i]:
                pop[i, :] = Xnew
                popCost[i] = newCost

                if newCost < BestCost:
                    BestCost = newCost
                    BestSol = Xnew

        BestCostArray[t] = BestCost

    return BestCost, BestSol, BestCostArray
