
import numpy as np

def FLA(NoMolecules, T, lb, ub, dim, objfunc):
    """
    Fick’s Law Algorithm (FLA)

    Parameters:
        NoMolecules (int): Population size.
        T (int): Maximum iterations.
        lb (float): Lower bound of variables.
        ub (float): Upper bound of variables.
        dim (int): Dimensionality of the problem.
        objfunc (function): Objective function.

    Returns:
        tuple: (Best fitness, Best solution, Convergence curve)
    """
    C1, C2, c3, c4, c5, D = 0.5, 2, 0.1, 0.2, 2, 0.01
    X = np.random.uniform(lb, ub, (NoMolecules, dim))
    CNVG = np.zeros(T)
    FS = np.array([objfunc(X[i, :]) for i in range(NoMolecules)])

    BestF, IndexBestF = np.min(FS), np.argmin(FS)
    Xss = X[IndexBestF, :].copy()

    n1, n2 = int(NoMolecules / 2), NoMolecules - int(NoMolecules / 2)
    X1, X2 = X[:n1, :], X[n1:, :]

    FS1, FS2 = np.array([objfunc(X1[i, :]) for i in range(n1)]), np.array([objfunc(X2[i, :]) for i in range(n2)])
    FSeo1, IndexFSeo1 = np.min(FS1), np.argmin(FS1)
    FSeo2, IndexFSeo2 = np.min(FS2), np.argmin(FS2)
    Xeo1, Xeo2 = X1[IndexFSeo1, :].copy(), X2[IndexFSeo2, :].copy()

    vec_flag = [1, -1]

    if FSeo1 < FSeo2:
        FSss = FSeo1
        YSol = Xeo1
    else:
        FSss = FSeo2
        YSol = Xeo2

    TF = np.sinh(np.arange(1, T + 1) / T) ** C1

    for t in range(T):
        if TF[t] < 0.9:
            DOF = np.exp(-(C2 * TF[t] - np.random.rand())) ** C2
            TDO = c5 * TF[t] - np.random.rand()
            if TDO < np.random.rand():
                NT12 = int((c4 * n1 - c3 * n1) * np.random.rand() + c3 * n1)
                for u in range(NT12):
                    DFg = np.random.choice(vec_flag)
                    Xm2, Xm1 = np.mean(X2, axis=0), np.mean(X1, axis=0)
                    J = -D * (Xm2 - Xm1) / (np.linalg.norm(Xeo2 - X1[u, :]) + np.finfo(float).eps)
                    X1[u, :] = Xeo2 + np.random.rand(dim) * DFg * DOF * (J * Xeo2 - X1[u, :])
        else:
            R1 = np.random.rand(dim)
            for u in range(n1):
                DFg = np.random.choice(vec_flag)
                Xm1, Xm = np.mean(X1, axis=0), np.mean(X, axis=0)
                J = -D * (Xm - Xm1) / (np.linalg.norm(Xss - X1[u, :]) + np.finfo(float).eps)
                DRF = np.exp(-J / TF[t])
                MS = np.exp(-FSss / (FS1[u] + np.finfo(float).eps))
                Qg = DFg * DRF * R1
                X1[u, :] = Xss + Qg * X1[u, :] + Qg * (MS * Xss - X1[u, :])

        FS1 = np.array([objfunc(X1[i, :]) for i in range(n1)])
        FS2 = np.array([objfunc(X2[i, :]) for i in range(n2)])

        FSeo1, IndexFSeo1 = np.min(FS1), np.argmin(FS1)
        FSeo2, IndexFSeo2 = np.min(FS2), np.argmin(FS2)
        Xeo1, Xeo2 = X1[IndexFSeo1, :], X2[IndexFSeo2, :]

        if FSeo1 < FSeo2:
            FSss = FSeo1
            YSol = Xeo1
        else:
            FSss = FSeo2
            YSol = Xeo2

        CNVG[t] = FSss

        if FSss < BestF:
            BestF = FSss
            Xss = YSol

    return BestF, Xss, CNVG
