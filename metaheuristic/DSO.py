
import numpy as np

def DSO(search_agent, run, LB, UB, dim, ObjFun):
    """
    Deep Sleep Optimizer (DSO)

    Parameters:
        search_agent (int): Population size.
        run (int): Maximum iterations.
        LB (float): Lower bound of variables.
        UB (float): Upper bound of variables.
        dim (int): Dimensionality of the problem.
        ObjFun (function): Objective function.

    Returns:
        tuple: (Best fitness, Best solution, Convergence curve)
    """
    prob = ObjFun
    nPop = search_agent
    MaxIt = run

    H0_minus = 0.17
    H0_plus = 0.85
    a = 0.1
    xs = 4.2
    xw = 18.2
    T = 24

    fit = np.zeros(nPop)
    Best_iteration = np.zeros(MaxIt + 1)

    C = np.sin((2 * np.pi) / T)
    H_min = H0_minus + a * C
    H_max = H0_plus + a * C

    Pop = np.random.uniform(LB, UB, (nPop, dim))

    for q in range(nPop):
        fit[q] = prob(Pop[q, :])
    Best_iteration[0] = np.min(fit)

    for i in range(MaxIt):
        for j in range(nPop):
            Xini = Pop[j, :]
            best_idx = np.argmin(fit)
            Xbest = Pop[best_idx, :]

            mu = np.random.rand()
            mu = np.clip(mu, H_min, H_max)
            H0 = Pop[j, :] + np.random.rand(dim) * (Xbest - mu * Xini)

            if np.random.rand() > mu:
                H = H0 * 10 ** (-1 / xs)
            else:
                H = mu + (H0 - mu) * 10 ** (-1 / xw)

            H = np.clip(H, LB, UB)

            fnew = prob(H)
            if fnew < fit[j]:
                Pop[j, :] = H
                fit[j] = fnew

        Best_iteration[i + 1] = np.min(fit)

    Best_Cost = np.min(fit)
    Best_Position = Pop[np.argmin(fit), :]

    return Best_Cost, Best_Position, Best_iteration
