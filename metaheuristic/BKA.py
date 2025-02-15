
import numpy as np

def BKA(pop, T, lb, ub, dim, fobj):
    XPos = lb + (ub - lb) * np.random.rand(pop, dim)
    XFit = np.array([fobj(XPos[i, :]) for i in range(pop)])
    Convergence_curve = np.zeros(T)

    Best_Fitness_BKA = np.inf
    Best_Pos_BKA = np.zeros(dim)

    for t in range(T):
        sorted_indexes = np.argsort(XFit)
        XLeader_Pos = XPos[sorted_indexes[0], :]
        XLeader_Fit = XFit[sorted_indexes[0]]

        for i in range(pop):
            n = 0.05 * np.exp(-2 * (t / T) ** 2)
            r = np.random.rand()

            if np.random.rand() < 0.9:
                XPosNew = XPos[i, :] + n * (1 + np.sin(r)) * XPos[i, :]
            else:
                XPosNew = XPos[i, :] * (n * (2 * np.random.rand(dim) - 1) + 1)

            XPosNew = np.clip(XPosNew, lb, ub)
            XFit_New = fobj(XPosNew)

            if XFit_New < XFit[i]:
                XPos[i, :] = XPosNew
                XFit[i] = XFit_New

        if np.min(XFit) < XLeader_Fit:
            Best_Fitness_BKA = np.min(XFit)
            Best_Pos_BKA = XPos[np.argmin(XFit), :]

        Convergence_curve[t] = Best_Fitness_BKA

    return Best_Fitness_BKA, Best_Pos_BKA, Convergence_curve
