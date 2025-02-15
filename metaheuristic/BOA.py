
import numpy as np

def BOA(n, N_iter, Lb, Ub, dim, fobj):
    p = 0.6
    power_exponent = 0.1
    sensory_modality = 0.01

    Sol = np.random.rand(n, dim) * (Ub - Lb) + Lb
    Fitness = np.array([fobj(Sol[i, :]) for i in range(n)])

    best_idx = np.argmin(Fitness)
    best_pos = Sol[best_idx, :].copy()

    Convergence_curve = np.zeros(N_iter)

    for t in range(N_iter):
        for i in range(n):
            FP = sensory_modality * (Fitness[i] ** power_exponent)

            if np.random.rand() < p:
                dis = np.random.rand() * best_pos - Sol[i, :]
                Sol[i, :] += dis * FP
            else:
                JK = np.random.permutation(n)
                dis = np.random.rand() * Sol[JK[0], :] - Sol[JK[1], :]
                Sol[i, :] += dis * FP

            Sol[i, :] = np.clip(Sol[i, :], Lb, Ub)
            Fitness[i] = fobj(Sol[i, :])

            if Fitness[i] < np.min(Fitness):
                best_pos = Sol[i, :].copy()

        Convergence_curve[t] = np.min(Fitness)

    return np.min(Fitness), best_pos, Convergence_curve
