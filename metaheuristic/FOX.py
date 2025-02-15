
import numpy as np

def FOX(SearchAgents_no, Max_iter, lb, ub, dim, fobj):
    """
    FOX: Fox-Inspired Optimization Algorithm

    Parameters:
        SearchAgents_no (int): Population size.
        Max_iter (int): Maximum iterations.
        lb (float): Lower bound of variables.
        ub (float): Upper bound of variables.
        dim (int): Dimensionality of the problem.
        fobj (function): Objective function.

    Returns:
        tuple: (Best fitness, Best solution, Convergence curve)
    """
    Best_pos = np.zeros(dim)
    Best_score = float('inf')
    MinT = float('inf')

    X = np.random.uniform(lb, ub, (SearchAgents_no, dim))
    Distance_Fox_Rat = np.zeros((SearchAgents_no, dim))
    convergence_curve = np.zeros(Max_iter)  

    l = 0  

    c1 = 0.18   
    c2 = 0.82   

    while l < Max_iter:
        for i in range(SearchAgents_no):
            X[i, :] = np.clip(X[i, :], lb, ub)
            fitness = fobj(X[i, :])

            if fitness < Best_score:
                Best_score = fitness
                Best_pos = X[i, :].copy()

        convergence_curve[l] = Best_score
        
        a = 2 * (1 - (l / Max_iter))
        Jump = 0.0

        for i in range(SearchAgents_no):
            r = np.random.rand()
            p = np.random.rand()

            if r >= 0.5:
                if p > 0.18:
                    Time = np.random.rand(dim)
                    sps = Best_pos / Time
                    Distance_S_Travel = sps * Time
                    Distance_Fox_Rat[i, :] = 0.5 * Distance_S_Travel
                    tt = np.sum(Time) / dim
                    t = tt / 2
                    Jump = 0.5 * 9.81 * t**2
                    X[i, :] = Distance_Fox_Rat[i, :] * Jump * c1

                elif p <= 0.18:
                    Time = np.random.rand(dim)
                    sps = Best_pos / Time
                    Distance_S_Travel = sps * Time
                    Distance_Fox_Rat[i, :] = 0.5 * Distance_S_Travel
                    tt = np.sum(Time) / dim
                    t = tt / 2
                    Jump = 0.5 * 9.81 * t**2
                    X[i, :] = Distance_Fox_Rat[i, :] * Jump * c2

                if MinT > tt:
                    MinT = tt

            elif r < 0.5:
                X[i, :] = Best_pos + np.random.randn(dim) * (MinT * a)

        l += 1

    return Best_score, Best_pos, convergence_curve
