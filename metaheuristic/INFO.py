import numpy as np
from scipy.special import gamma



def info(nP, max_it, lb, ub, dim, fobj):
    cost = np.zeros(nP)
    M = np.zeros(nP)

    X = np.random.uniform(lb, ub, (nP, dim))

    for i in range(nP):
        cost[i] = fobj(X[i, :])
        M[i] = cost[i]

    sorted_inds = np.argsort(cost)
    Best_X = X[sorted_inds[0], :]
    Best_Cost = cost[sorted_inds[0]]

    Worst_Cost = cost[sorted_inds[-1]]
    Worst_X = X[sorted_inds[-1], :]

    I = np.random.randint(2, 6)
    Better_X = X[sorted_inds[I], :]
    Better_Cost = cost[sorted_inds[I]]

    convergence_curve = np.zeros(max_it)

    for it in range(max_it):
        alpha = 2 * np.exp(-4 * (it / max_it))
        M_Best = Best_Cost
        M_Better = Better_Cost
        M_Worst = Worst_Cost

        for i in range(nP):
            del_val = 2 * np.random.rand() * alpha - alpha
            sigm = 2 * np.random.rand() * alpha - alpha

            A1 = np.random.permutation(nP)
            A1 = A1[A1 != i]
            a, b, c = A1[:3]

            e = 1e-25
            epsi = e * np.random.rand()

            omg = max([M[a], M[b], M[c]])
            MM = [M[a] - M[b], M[a] - M[c], M[b] - M[c]]

            W = np.cos(np.array(MM) + np.pi) * np.exp(-np.array(MM) / omg)
            Wt = np.sum(W)

            WM1 = del_val * (W[0] * (X[a, :] - X[b, :]) + W[1] * (X[a, :] - X[c, :]) + W[2] * (X[b, :] - X[c, :])) / (
                        Wt + 1) + epsi

            omg = max([M_Best, M_Better, M_Worst])
            MM = [M_Best - M_Better, M_Best - M_Worst, M_Better - M_Worst]

            W = np.cos(np.array(MM) + np.pi) * np.exp(-np.array(MM) / omg)
            Wt = np.sum(W)

            WM2 = del_val * (W[0] * (Best_X - Better_X) + W[1] * (Best_X - Worst_X) + W[2] * (Better_X - Worst_X)) / (
                        Wt + 1) + epsi

            r = np.random.uniform(0.1, 0.5)
            MeanRule = r * WM1 + (1 - r) * WM2

            if np.random.rand() < 0.5:
                z1 = X[i, :] + sigm * (np.random.rand() * MeanRule) + np.random.randn() * (Best_X - X[a, :]) / (
                            M_Best - M[a] + 1)
                z2 = Best_X + sigm * (np.random.rand() * MeanRule) + np.random.randn() * (X[a, :] - X[b, :]) / (
                            M[a] - M[b] + 1)
            else:
                z1 = X[a, :] + sigm * (np.random.rand() * MeanRule) + np.random.randn() * (X[b, :] - X[c, :]) / (
                            M[b] - M[c] + 1)
                z2 = Better_X + sigm * (np.random.rand() * MeanRule) + np.random.randn() * (X[a, :] - X[b, :]) / (
                            M[a] - M[b] + 1)

            u = np.zeros(dim)
            for j in range(dim):
                mu = 0.05 * np.random.randn()
                if np.random.rand() < 0.5:
                    if np.random.rand() < 0.5:
                        u[j] = z1[j] + mu * abs(z1[j] - z2[j])
                    else:
                        u[j] = z2[j] + mu * abs(z1[j] - z2[j])
                else:
                    u[j] = X[i, j]

            New_X = np.clip(u, lb, ub)
            New_Cost = fobj(New_X)

            if New_Cost < cost[i]:
                X[i, :] = New_X
                cost[i] = New_Cost
                M[i] = cost[i]
                if cost[i] < Best_Cost:
                    Best_X = X[i, :]
                    Best_Cost = cost[i]

        sorted_inds = np.argsort(cost)
        Worst_X = X[sorted_inds[-1], :]
        Worst_Cost = cost[sorted_inds[-1]]
        I = np.random.randint(2, 6)
        Better_X = X[sorted_inds[I], :]
        Better_Cost = cost[sorted_inds[I]]

        convergence_curve[it] = Best_Cost

    return Best_Cost, Best_X, convergence_curve