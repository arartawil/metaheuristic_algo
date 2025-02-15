import numpy as np
from scipy.special import gamma


def levy_flight_distribution(n, max_iter, lb, ub, dim, fobj):
    threshold = 2
    lb = np.full(dim, lb)
    ub = np.full(dim, ub)

    X = np.random.uniform(lb, ub, (n, dim))
    X_fitness = np.array([fobj(X[i, :]) for i in range(n)])

    best_idx = np.argmin(X_fitness)
    target_position = X[best_idx, :]
    target_fitness = X_fitness[best_idx]

    convergence_curve = np.zeros(max_iter)
    vec_flag = np.array([1, -1])

    for l in range(max_iter):
        D = np.zeros(n)

        for i in range(n):
            X[i, :] = np.clip(X[i, :], lb, ub)

            neighbors = []
            neighbor_count = 0

            for j in range(n):
                if i != j:
                    distance = np.linalg.norm(X[j, :] - X[i, :])
                    if distance < threshold:
                        temp = X_fitness[j] / (X_fitness[i] + 1e-10)
                        temp = (0.9 * (temp - np.min(temp))) / (np.max(temp) - np.min(temp) + 1e-10) + 0.1
                        neighbor_count += 1
                        D[neighbor_count - 1] = temp
                        neighbors.append(X[j, :])

            S_i = np.zeros(dim)
            for p in range(neighbor_count):
                flag_index = np.random.randint(0, 2)
                var_flag = vec_flag[flag_index]
                S_i += var_flag * D[p] * neighbors[p] / neighbor_count

            rand_leader_index = np.random.randint(0, n)
            X_rand = X[rand_leader_index, :]
            X_new = target_position + 10 * S_i + np.random.rand() * 0.00005 * (
                        (target_position + 0.005 * X_rand) / 2 - X[i, :])
            X_new = levy_flight(X_new, target_position, dim)

            X[i, :] = X_new
            X_fitness[i] = fobj(X[i, :])

        best_idx = np.argmin(X_fitness)
        if X_fitness[best_idx] < target_fitness:
            target_position = X[best_idx, :]
            target_fitness = X_fitness[best_idx]

        convergence_curve[l] = target_fitness

    return target_fitness, target_position, convergence_curve


def levy_flight(pos, pos_target, dim):
    beta = 1.5
    sigma = (gamma(1 + beta) * np.sin(np.pi * beta / 2) / (gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (
                1 / beta)

    for j in range(dim):
        u = np.random.rand() * sigma
        v = np.random.rand()
        step = u / np.abs(v) ** (1 / beta)
        step_size = 0.01 * step * (pos[j] - pos_target[j])
        pos[j] += step_size * np.random.rand()

    return pos
