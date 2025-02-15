
import numpy as np

def AOArithmetic(N, M_Iter, LB, UB, Dim, F_obj):
    Best_P = np.zeros(Dim)
    Best_FF = np.inf
    Conv_curve = np.zeros(M_Iter)

    X = np.random.rand(N, Dim) * (UB - LB) + LB
    Xnew = X.copy()
    Ffun = np.zeros(N)
    Ffun_new = np.zeros(N)

    MOP_Max = 1
    MOP_Min = 0.2
    C_Iter = 1
    Alpha = 5
    Mu = 0.499

    for i in range(N):
        Ffun[i] = F_obj(X[i, :])
        if Ffun[i] < Best_FF:
            Best_FF = Ffun[i]
            Best_P = X[i, :]

    while C_Iter <= M_Iter:
        MOP = 1 - ((C_Iter)**(1/Alpha) / (M_Iter)**(1/Alpha))
        MOA = MOP_Min + C_Iter * ((MOP_Max - MOP_Min) / M_Iter)

        for i in range(N):
            for j in range(Dim):
                r1 = np.random.rand()
                if r1 < MOA:
                    r2 = np.random.rand()
                    if r2 > 0.5:
                        Xnew[i, j] = Best_P[j] / (MOP + np.finfo(float).eps) * ((UB - LB) * Mu + LB)
                    else:
                        Xnew[i, j] = Best_P[j] * MOP * ((UB - LB) * Mu + LB)
                else:
                    r3 = np.random.rand()
                    if r3 > 0.5:
                        Xnew[i, j] = Best_P[j] - MOP * ((UB - LB) * Mu + LB)
                    else:
                        Xnew[i, j] = Best_P[j] + MOP * ((UB - LB) * Mu + LB)

            Xnew[i, :] = np.clip(Xnew[i, :], LB, UB)
            Ffun_new[i] = F_obj(Xnew[i, :])
            if Ffun_new[i] < Ffun[i]:
                X[i, :] = Xnew[i, :]
                Ffun[i] = Ffun_new[i]

            if Ffun[i] < Best_FF:
                Best_FF = Ffun[i]
                Best_P = X[i, :]

        Conv_curve[C_Iter - 1] = Best_FF
        C_Iter += 1

    return Best_FF, Best_P, Conv_curve
