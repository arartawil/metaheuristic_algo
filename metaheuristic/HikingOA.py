import numpy as np
from scipy.special import gamma


def levy(d):
    beta = 1.5
    sigma = (gamma(1 + beta) * np.sin(np.pi * beta / 2) / (gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (
                1 / beta)
    u = np.random.randn(d) * sigma
    v = np.random.randn(d)
    step = u / np.abs(v) ** (1 / beta)
    return step


def hiking_oa(hiker, max_iter, lb, ub, dim, obj_func):
    fit = np.zeros(hiker)
    best_iteration = np.zeros(max_iter + 1)

    Pop = np.array([lb + (ub - lb) * np.random.rand(dim) for _ in range(hiker)])

    for q in range(hiker):
        fit[q] = obj_func(Pop[q])

    best_iteration[0] = np.min(fit)

    for i in range(max_iter):
        ind = np.argmin(fit)
        Xbest = Pop[ind]

        for j in range(hiker):
            Xini = Pop[j]
            theta = np.random.randint(0, 51)
            s = np.tan(np.radians(theta))
            SF = np.random.choice([1, 2])

            Vel = 6 * np.exp(-3.5 * abs(s + 0.05))
            newVel = Vel + np.random.rand(dim) * (Xbest - SF * Xini)
            newPop = Xini + newVel
            newPop = np.clip(newPop, lb, ub)

            fnew = obj_func(newPop)
            if fnew < fit[j]:
                Pop[j] = newPop
                fit[j] = fnew

        best_iteration[i + 1] = np.min(fit)

    best_hike, idx = np.min(fit), np.argmin(fit)
    best_position = Pop[idx]

    return best_hike, best_position, best_iteration
