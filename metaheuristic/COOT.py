
import numpy as np

def COOT(N, Max_iter, lb, ub, dim, fobj):
    NLeader = int(np.ceil(0.1 * N))
    Ncoot = N - NLeader
    Convergence_curve = np.zeros(Max_iter)
    gBest = np.zeros(dim)
    gBestScore = np.inf

    CootPos = np.random.rand(Ncoot, dim) * (ub - lb) + lb
    CootFitness = np.array([fobj(CootPos[i, :]) for i in range(Ncoot)])

    LeaderPos = np.random.rand(NLeader, dim) * (ub - lb) + lb
    LeaderFit = np.array([fobj(LeaderPos[i, :]) for i in range(NLeader)])

    for i in range(Ncoot):
        if gBestScore > CootFitness[i]:
            gBestScore = CootFitness[i]
            gBest = CootPos[i, :]

    for i in range(NLeader):
        if gBestScore > LeaderFit[i]:
            gBestScore = LeaderFit[i]
            gBest = LeaderPos[i, :]

    Convergence_curve[0] = gBestScore

    for l in range(1, Max_iter):
        B = 2 - l * (1 / Max_iter)
        A = 1 - l * (1 / Max_iter)

        for i in range(Ncoot):
            R = -1 + 2 * np.random.rand()
            R1 = np.random.rand()

            k = i % NLeader
            if np.random.rand() < 0.5:
                CootPos[i, :] = 2 * R1 * np.cos(2 * np.pi * R) * (LeaderPos[k, :] - CootPos[i, :]) + LeaderPos[k, :]
            else:
                if np.random.rand() < 0.5 and i != 0:
                    CootPos[i, :] = (CootPos[i, :] + CootPos[i - 1, :]) / 2
                else:
                    Q = np.random.rand(dim) * (ub - lb) + lb
                    CootPos[i, :] = CootPos[i, :] + A * R1 * (Q - CootPos[i, :])

            CootPos[i, :] = np.clip(CootPos[i, :], lb, ub)

        CootFitness = np.array([fobj(CootPos[i, :]) for i in range(Ncoot)])

        for i in range(Ncoot):
            k = i % NLeader
            if CootFitness[i] < LeaderFit[k]:
                LeaderFit[k], CootFitness[i] = CootFitness[i], LeaderFit[k]
                LeaderPos[k, :], CootPos[i, :] = CootPos[i, :], LeaderPos[k, :]

        for i in range(NLeader):
            R = -1 + 2 * np.random.rand()
            R3 = np.random.rand()
            Temp = B * R3 * np.cos(2 * np.pi * R) * (gBest - LeaderPos[i, :]) + gBest
            Temp = np.clip(Temp, lb, ub)
            TempFit = fobj(Temp)

            if gBestScore > TempFit:
                LeaderFit[i] = gBestScore
                LeaderPos[i, :] = gBest
                gBestScore = TempFit
                gBest = Temp

        Convergence_curve[l] = gBestScore

    return gBestScore, gBest, Convergence_curve
