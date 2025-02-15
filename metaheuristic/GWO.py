import numpy as np

def initialization(pop_size, dim, ub, lb):
    return lb + np.random.rand(pop_size, dim) * (ub - lb)

def GWO(SearchAgents_no, Max_iter, lb, ub, dim, fobj):
    Alpha_pos = np.zeros(dim)
    Alpha_score = float('inf')

    Beta_pos = np.zeros(dim)
    Beta_score = float('inf')

    Delta_pos = np.zeros(dim)
    Delta_score = float('inf')

    Positions = initialization(SearchAgents_no, dim, ub, lb)
    Convergence_curve = np.zeros(Max_iter)

    for l in range(Max_iter):
        for i in range(SearchAgents_no):
            Positions[i, :] = np.clip(Positions[i, :], lb, ub)
            fitness = fobj(Positions[i, :])

            if fitness < Alpha_score:
                Alpha_score = fitness
                Alpha_pos = Positions[i, :]

            if Alpha_score < fitness < Beta_score:
                Beta_score = fitness
                Beta_pos = Positions[i, :]

            if Alpha_score < fitness < Beta_score < fitness < Delta_score:
                Delta_score = fitness
                Delta_pos = Positions[i, :]

        a = 2 - l * (2 / Max_iter)

        for i in range(SearchAgents_no):
            for j in range(dim):
                r1, r2 = np.random.rand(), np.random.rand()
                A1 = 2 * a * r1 - a
                C1 = 2 * r2
                D_alpha = abs(C1 * Alpha_pos[j] - Positions[i, j])
                X1 = Alpha_pos[j] - A1 * D_alpha

                r1, r2 = np.random.rand(), np.random.rand()
                A2 = 2 * a * r1 - a
                C2 = 2 * r2
                D_beta = abs(C2 * Beta_pos[j] - Positions[i, j])
                X2 = Beta_pos[j] - A2 * D_beta

                r1, r2 = np.random.rand(), np.random.rand()
                A3 = 2 * a * r1 - a
                C3 = 2 * r2
                D_delta = abs(C3 * Delta_pos[j] - Positions[i, j])
                X3 = Delta_pos[j] - A3 * D_delta

                Positions[i, j] = (X1 + X2 + X3) / 3

        Convergence_curve[l] = Alpha_score

    return Alpha_score, Alpha_pos, Convergence_curve
