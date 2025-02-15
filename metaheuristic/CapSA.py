
import numpy as np

def CapSA(noP, maxite, LB, UB, dim, fobj):
    CapPos = np.random.rand(noP, dim) * (UB - LB) + LB
    CapFit = np.array([fobj(CapPos[i, :]) for i in range(noP)])

    fitCapSA = np.min(CapFit)
    gFoodPos = CapPos[np.argmin(CapFit), :]

    Convergence_curve = np.zeros(maxite)

    for t in range(maxite):
        for i in range(noP):
            newPos = CapPos[i, :] + np.random.rand(dim) * (gFoodPos - CapPos[i, :])
            newPos = np.clip(newPos, LB, UB)
            newFit = fobj(newPos)

            if newFit < CapFit[i]:
                CapPos[i, :] = newPos
                CapFit[i] = newFit

            if newFit < fitCapSA:
                gFoodPos = newPos.copy()
                fitCapSA = newFit

        Convergence_curve[t] = fitCapSA

    return fitCapSA, gFoodPos, Convergence_curve
