
import numpy as np

def AHA(nPop, MaxIt, Low, Up, Dim, FunIndex):
    PopPos = np.random.rand(nPop, Dim) * (Up - Low) + Low
    PopFit = np.array([FunIndex(PopPos[i, :]) for i in range(nPop)])

    BestF = np.min(PopFit)
    BestX = PopPos[np.argmin(PopFit), :]

    HisBestFit = np.zeros(MaxIt)

    for It in range(MaxIt):
        newPopPos = np.zeros_like(PopPos)

        for i in range(nPop):
            if np.random.rand() < 0.5:
                newPopPos[i, :] = PopPos[i, :] + np.random.randn(Dim) * PopPos[i, :]
            else:
                TargetFoodIndex = np.argmax(PopFit)
                newPopPos[i, :] = PopPos[TargetFoodIndex, :] + np.random.randn() * (PopPos[i, :] - PopPos[TargetFoodIndex, :])

            newPopPos[i, :] = np.clip(newPopPos[i, :], Low, Up)
            newPopFit = FunIndex(newPopPos[i, :])

            if newPopFit < PopFit[i]:
                PopFit[i] = newPopFit
                PopPos[i, :] = newPopPos[i, :]

        if np.min(PopFit) < BestF:
            BestF = np.min(PopFit)
            BestX = PopPos[np.argmin(PopFit), :]

        HisBestFit[It] = BestF

    return BestF, BestX, HisBestFit
