
import numpy as np

def CSBO(nPop, MaxIt, VarMin, VarMax, nVar, CostFunction):
    BestSolPosition = np.zeros(nVar)
    BestSolCost = np.inf

    pop = np.random.rand(nPop, nVar) * (VarMax - VarMin) + VarMin
    Cost = np.array([CostFunction(pop[i, :]) for i in range(nPop)])
    BestIndex = np.argmin(Cost)
    BestSolPosition = pop[BestIndex, :].copy()
    BestSolCost = Cost[BestIndex]

    BestCost = np.zeros(MaxIt)

    for It in range(MaxIt):
        for i in range(nPop):
            A = np.random.permutation(nPop)
            A = A[A != i]

            a1, a2, a3 = A[:3]
            AA1 = (Cost[a2] - Cost[a3]) / np.abs(Cost[a3] - Cost[a2])
            AA2 = (Cost[a1] - Cost[i]) / np.abs(Cost[a1] - Cost[i])

            pos = pop[i, :] + AA2 * np.random.rand(nVar) * (pop[i, :] - pop[a1, :]) +                 AA1 * np.random.rand(nVar) * (pop[a3, :] - pop[a2, :])

            pos = np.clip(pos, VarMin, VarMax)
            cost = CostFunction(pos)

            if cost < Cost[i]:
                pop[i, :] = pos
                Cost[i] = cost

                if cost < BestSolCost:
                    BestSolPosition = pos.copy()
                    BestSolCost = cost

        BestCost[It] = BestSolCost

    return BestSolCost, BestSolPosition, BestCost
