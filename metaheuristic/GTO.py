import numpy as np

def initialization(pop_size, dim, ub, lb):
    return lb + np.random.rand(pop_size, dim) * (ub - lb)

def GTO(pop_size, max_iter, lower_bound, upper_bound, variables_no, fobj):
    Silverback = np.zeros(variables_no)
    Silverback_Score = float('inf')

    X = initialization(pop_size, variables_no, upper_bound, lower_bound)
    convergence_curve = np.zeros(max_iter)

    Pop_Fit = np.zeros(pop_size)
    for i in range(pop_size):   
        Pop_Fit[i] = fobj(X[i, :])
        if Pop_Fit[i] < Silverback_Score:
            Silverback_Score = Pop_Fit[i]
            Silverback = X[i, :]

    GX = X.copy()
    lb = np.full(variables_no, lower_bound)
    ub = np.full(variables_no, upper_bound)

    p = 0.03
    Beta = 3
    w = 0.8

    for It in range(max_iter):
        a = (np.cos(2 * np.random.rand()) + 1) * (1 - It / max_iter)
        C = a * (2 * np.random.rand() - 1)

        for i in range(pop_size):
            if np.random.rand() < p:
                GX[i, :] = lb + np.random.rand(variables_no) * (ub - lb)
            else:
                if np.random.rand() >= 0.5:
                    Z = np.random.rand(variables_no) * (2 * a) - a
                    H = Z * X[i, :]
                    GX[i, :] = (np.random.rand() - a) * X[np.random.randint(pop_size), :] + C * H
                else:
                    GX[i, :] = X[i, :] - C * (C * (X[i, :] - GX[np.random.randint(pop_size), :]) + np.random.rand() * (X[i, :] - GX[np.random.randint(pop_size), :]))

        GX = np.clip(GX, lower_bound, upper_bound)

        for i in range(pop_size):
            New_Fit = fobj(GX[i, :])
            if New_Fit < Pop_Fit[i]:
                Pop_Fit[i] = New_Fit
                X[i, :] = GX[i, :]
            if New_Fit < Silverback_Score:
                Silverback_Score = New_Fit
                Silverback = GX[i, :]

        for i in range(pop_size):
            if a >= w:
                g = 2**C
                delta = (np.abs(np.mean(GX, axis=0)) ** g) ** (1/g)
                GX[i, :] = C * delta * (X[i, :] - Silverback) + X[i, :]
            else:
                h = np.random.randn() if np.random.rand() >= 0.5 else np.random.randn()
                r1 = np.random.rand()
                GX[i, :] = Silverback - (Silverback * (2 * r1 - 1) - X[i, :] * (2 * r1 - 1)) * (Beta * h)

        GX = np.clip(GX, lower_bound, upper_bound)

        for i in range(pop_size):
            New_Fit = fobj(GX[i, :])
            if New_Fit < Pop_Fit[i]:
                Pop_Fit[i] = New_Fit
                X[i, :] = GX[i, :]
            if New_Fit < Silverback_Score:
                Silverback_Score = New_Fit
                Silverback = GX[i, :]

        convergence_curve[It] = Silverback_Score

    return Silverback_Score, Silverback, convergence_curve
