
import numpy as np

def AEO(nPop, MaxIt, Low, Up, Dim, F_index):
    PopPos = np.random.rand(nPop, Dim) * (Up - Low) + Low
    PopFit = np.array([F_index(PopPos[i, :]) for i in range(nPop)])

    BestF = np.min(PopFit)
    BestX = PopPos[np.argmin(PopFit), :]

    HisBestFit = np.zeros(MaxIt)

    for It in range(MaxIt):
        r1 = np.random.rand()
        a = (1 - It / MaxIt) * r1
        xrand = np.random.rand(Dim) * (Up - Low) + Low
        newPopPos = np.zeros((nPop, Dim))
        newPopPos[0, :] = (1 - a) * PopPos[-1, :] + a * xrand

        for i in range(2, nPop):
            u, v = np.random.randn(Dim), np.random.randn(Dim)
            C = 0.5 * u / np.abs(v)
            r = np.random.rand()

            if r < 1 / 3:
                newPopPos[i, :] = PopPos[i, :] + C * (PopPos[i, :] - newPopPos[0, :])
            elif r < 2 / 3:
                r_idx = np.random.randint(1, i)
                newPopPos[i, :] = PopPos[i, :] + C * (PopPos[i, :] - PopPos[r_idx, :])
            else:
                r2 = np.random.rand()
                r_idx = np.random.randint(1, i)
                newPopPos[i, :] = PopPos[i, :] + C * (r2 * (PopPos[i, :] - newPopPos[0, :]) + (1 - r2) * (PopPos[i, :] - PopPos[r_idx, :]))

        newPopPos = np.clip(newPopPos, Low, Up)
        newPopFit = np.array([F_index(newPopPos[i, :]) for i in range(nPop)])

        for i in range(nPop):
            if newPopFit[i] < PopFit[i]:
                PopPos[i, :] = newPopPos[i, :]
                PopFit[i] = newPopFit[i]

        if np.min(PopFit) < BestF:
            BestF = np.min(PopFit)
            BestX = PopPos[np.argmin(PopFit), :]

        HisBestFit[It] = BestF

    return BestF, BestX, HisBestFit
