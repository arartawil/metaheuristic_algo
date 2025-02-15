
import numpy as np

def CDO(SearchAgents_no, Max_iter, lb, ub, dim, fobj):
    Alpha_pos = np.zeros(dim)
    Alpha_score = np.inf

    Beta_pos = np.zeros(dim)
    Beta_score = np.inf

    Gamma_pos = np.zeros(dim)
    Gamma_score = np.inf

    Positions = np.random.rand(SearchAgents_no, dim) * (ub - lb) + lb
    Convergence_curve = np.zeros(Max_iter)

    for l in range(Max_iter):
        for i in range(SearchAgents_no):
            Positions[i, :] = np.clip(Positions[i, :], lb, ub)
            fitness = fobj(Positions[i, :])

            if fitness < Alpha_score:
                Alpha_score = fitness
                Alpha_pos = Positions[i, :].copy()
            elif fitness > Alpha_score and fitness < Beta_score:
                Beta_score = fitness
                Beta_pos = Positions[i, :].copy()
            elif fitness > Alpha_score and fitness > Beta_score and fitness < Gamma_score:
                Gamma_score = fitness
                Gamma_pos = Positions[i, :].copy()

        a = 3 - l * (3 / Max_iter)

        for i in range(SearchAgents_no):
            for j in range(dim):
                r1, r2 = np.random.rand(), np.random.rand()
                D_alpha = abs(r2 * Alpha_pos[j] - Positions[i, j])
                D_beta = abs(r2 * Beta_pos[j] - Positions[i, j])
                D_gamma = abs(r2 * Gamma_pos[j] - Positions[i, j])

                Positions[i, j] = (Alpha_pos[j] - a * D_alpha + 
                                   Beta_pos[j] - a * D_beta + 
                                   Gamma_pos[j] - a * D_gamma) / 3

        Convergence_curve[l] = Alpha_score

    return Alpha_score, Alpha_pos, Convergence_curve
