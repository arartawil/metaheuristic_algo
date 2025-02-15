import numpy as np

def initialization(pop_size, dim, ub, lb):
    return lb + np.random.rand(pop_size, dim) * (ub - lb)

def gradient_search_rule(ro1, Best_X, Worst_X, X, Xr1, DM, eps, Xm, Flag):
    nV = len(X)
    Delta = 2 * np.random.rand(nV) * np.abs(Xm - X)
    Step = ((Best_X - Xr1) + Delta) / 2
    DelX = np.random.rand(nV) * np.abs(Step)
    
    GSR = np.random.randn() * ro1 * (2 * DelX * X) / (Best_X - Worst_X + eps)
    
    if Flag == 1:
        Xs = X - GSR + DM
    else:
        Xs = Best_X - GSR + DM

    yp = np.random.rand() * (0.5 * (Xs + X) + np.random.rand() * DelX)
    yq = np.random.rand() * (0.5 * (Xs + X) - np.random.rand() * DelX)
    
    GSR = np.random.randn() * ro1 * (2 * DelX * X) / (yp - yq + eps)
    return GSR

def GBO(nP, MaxIt, lb, ub, dim, fobj):
    pr = 0.5
    lb = np.full(dim, lb)
    ub = np.full(dim, ub)
    Cost = np.zeros(nP)
    X = initialization(nP, dim, ub, lb)
    Convergence_curve = np.zeros(MaxIt)

    for i in range(nP):
        Cost[i] = fobj(X[i, :])

    sort_idx = np.argsort(Cost)
    Best_Cost = Cost[sort_idx[0]]
    Best_X = X[sort_idx[0], :]
    Worst_Cost = Cost[sort_idx[-1]]
    Worst_X = X[sort_idx[-1], :]

    for it in range(MaxIt):
        beta = 0.2 + (1.2 - 0.2) * (1 - (it / MaxIt) ** 3) ** 2
        alpha = np.abs(beta * np.sin(3 * np.pi / 2 + np.sin(3 * np.pi / 2 * beta)))

        for i in range(nP):
            A1 = np.random.randint(0, nP, 4)
            r1, r2, r3, r4 = A1

            Xm = (X[r1, :] + X[r2, :] + X[r3, :] + X[r4, :]) / 4
            ro = alpha * (2 * np.random.rand() - 1)
            ro1 = alpha * (2 * np.random.rand() - 1)
            eps = 5e-3 * np.random.rand()

            DM = np.random.rand() * ro * (Best_X - X[r1, :])
            Flag = 1
            GSR = gradient_search_rule(ro1, Best_X, Worst_X, X[i, :], X[r1, :], DM, eps, Xm, Flag)
            DM = np.random.rand() * ro * (Best_X - X[r1, :])
            X1 = X[i, :] - GSR + DM

            DM = np.random.rand() * ro * (X[r1, :] - X[r2, :])
            Flag = 2
            GSR = gradient_search_rule(ro1, Best_X, Worst_X, X[i, :], X[r1, :], DM, eps, Xm, Flag)
            DM = np.random.rand() * ro * (X[r1, :] - X[r2, :])
            X2 = Best_X - GSR + DM

            Xnew = np.zeros(dim)
            for j in range(dim):
                ro = alpha * (2 * np.random.rand() - 1)
                X3 = X[i, j] - ro * (X2[j] - X1[j])
                ra, rb = np.random.rand(), np.random.rand()
                Xnew[j] = ra * (rb * X1[j] + (1 - rb) * X2[j]) + (1 - ra) * X3

            if np.random.rand() < pr:
                k = np.random.randint(0, nP)
                f1 = -1 + 2 * np.random.rand()
                f2 = -1 + 2 * np.random.rand()
                ro = alpha * (2 * np.random.rand() - 1)
                Xk = lb + (ub - lb) * np.random.rand(dim)

                L1 = np.random.rand() < 0.5
                u1 = L1 * 2 * np.random.rand() + (1 - L1)
                u2 = L1 * np.random.rand() + (1 - L1)
                u3 = L1 * np.random.rand() + (1 - L1)
                L2 = np.random.rand() < 0.5
                Xp = (1 - L2) * X[k, :] + L2 * Xk

                if u1 < 0.5:
                    Xnew = Xnew + f1 * (u1 * Best_X - u2 * Xp) + f2 * ro * (u3 * (X2 - X1) + u2 * (X[r1, :] - X[r2, :])) / 2
                else:
                    Xnew = Best_X + f1 * (u1 * Best_X - u2 * Xp) + f2 * ro * (u3 * (X2 - X1) + u2 * (X[r1, :] - X[r2, :])) / 2

            Xnew = np.clip(Xnew, lb, ub)
            Xnew_Cost = fobj(Xnew)

            if Xnew_Cost < Cost[i]:
                X[i, :] = Xnew
                Cost[i] = Xnew_Cost
                if Cost[i] < Best_Cost:
                    Best_X = X[i, :]
                    Best_Cost = Cost[i]

            if Cost[i] > Worst_Cost:
                Worst_X = X[i, :]
                Worst_Cost = Cost[i]

        Convergence_curve[it] = Best_Cost

    return Best_Cost, Best_X, Convergence_curve
