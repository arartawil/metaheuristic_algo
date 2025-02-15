
import numpy as np

def ARO(nPop, MaxIt, Low, Up, Dim, F_index):
    PopPos = Low + np.random.rand(nPop, Dim) * (Up - Low)
    PopFit = np.array([F_index(PopPos[i, :]) for i in range(nPop)])

    BestF = np.inf
    BestX = np.zeros(Dim)

    for i in range(nPop):
        if PopFit[i] < BestF:
            BestF = PopFit[i]
            BestX = PopPos[i, :]

    HisBestF = np.zeros(MaxIt)

    for It in range(MaxIt):
        Direct1 = np.zeros((nPop, Dim))
        Direct2 = np.zeros((nPop, Dim))
        theta = 2 * (1 - It / MaxIt)

        for i in range(nPop):
            L = (np.exp(1) - np.exp(((It - 1) / MaxIt) ** 2)) * np.sin(2 * np.pi * np.random.rand())
            Direct1[i, np.random.choice(Dim, size=int(np.ceil(np.random.rand() * Dim)), replace=False)] = 1
            R = L * Direct1[i, :]

            A = 2 * np.log(1 / np.random.rand()) * theta

            if A > 1:
                RandInd = np.random.randint(0, nPop)
                newPopPos = PopPos[RandInd, :] + R * (PopPos[i, :] - PopPos[RandInd, :]) + np.round(0.5 * (0.05 + np.random.rand())) * np.random.randn(Dim)
            else:
                Direct2[i, np.random.randint(0, Dim)] = 1
                gr = Direct2[i, :]
                H = ((MaxIt - It + 1) / MaxIt) * np.random.randn()
                b = PopPos[i, :] + H * gr * PopPos[i, :]
                newPopPos = PopPos[i, :] + R * (np.random.rand() * b - PopPos[i, :])

            newPopPos = np.clip(newPopPos, Low, Up)
            newPopFit = F_index(newPopPos)

            if newPopFit < PopFit[i]:
                PopFit[i] = newPopFit
                PopPos[i, :] = newPopPos

        BestF = np.min(PopFit)
        BestX = PopPos[np.argmin(PopFit), :]
        HisBestF[It] = BestF

    return BestF, BestX, HisBestF
