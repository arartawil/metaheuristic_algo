
import numpy as np

def BES(nPop, MaxIt, low, high, dim, fobj):
    BestCost = np.inf
    BestPos = np.zeros(dim)

    popPos = low + (high - low) * np.random.rand(nPop, dim)
    popCost = np.array([fobj(popPos[i, :]) for i in range(nPop)])

    for i in range(nPop):
        if popCost[i] < BestCost:
            BestPos = popPos[i, :]
            BestCost = popCost[i]

    Convergence_curve = np.zeros(MaxIt)

    for t in range(MaxIt):
        # Select space operation
        Mean = np.mean(popPos, axis=0)
        lm = 2

        for i in range(nPop):
            newsolPos = BestPos + lm * np.random.rand(dim) * (Mean - popPos[i, :])
            newsolPos = np.clip(newsolPos, low, high)
            newsolCost = fobj(newsolPos)

            if newsolCost < popCost[i]:
                popPos[i, :] = newsolPos
                popCost[i] = newsolCost

                if popCost[i] < BestCost:
                    BestPos = popPos[i, :]
                    BestCost = popCost[i]

        Convergence_curve[t] = BestCost

    return BestCost, BestPos, Convergence_curve
