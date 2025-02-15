
import numpy as np

def ChOA(SearchAgents_no, Max_iter, lb, ub, dim, fobj):
    Attacker_pos = np.zeros(dim)
    Attacker_score = np.inf

    Barrier_pos = np.zeros(dim)
    Barrier_score = np.inf

    Chaser_pos = np.zeros(dim)
    Chaser_score = np.inf

    Driver_pos = np.zeros(dim)
    Driver_score = np.inf

    Positions = np.random.rand(SearchAgents_no, dim) * (ub - lb) + lb

    Convergence_curve = np.zeros(Max_iter)

    for l in range(Max_iter):
        for i in range(SearchAgents_no):
            Positions[i, :] = np.clip(Positions[i, :], lb, ub)
            fitness = fobj(Positions[i, :])

            if fitness < Attacker_score:
                Attacker_score = fitness
                Attacker_pos = Positions[i, :].copy()
            elif fitness > Attacker_score and fitness < Barrier_score:
                Barrier_score = fitness
                Barrier_pos = Positions[i, :].copy()
            elif fitness > Attacker_score and fitness > Barrier_score and fitness < Chaser_score:
                Chaser_score = fitness
                Chaser_pos = Positions[i, :].copy()
            elif fitness > Attacker_score and fitness > Barrier_score and fitness > Chaser_score and fitness < Driver_score:
                Driver_score = fitness
                Driver_pos = Positions[i, :].copy()

        f = 2 - l * (2 / Max_iter)

        for i in range(SearchAgents_no):
            Positions[i, :] = (Attacker_pos + Barrier_pos + Chaser_pos + Driver_pos) / 4

        Convergence_curve[l] = Attacker_score

    return Attacker_score, Attacker_pos, Convergence_curve
