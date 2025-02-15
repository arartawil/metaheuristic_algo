import numpy as np


def lungs_performance_optimization(n_pop, max_it, var_min, var_max, n_var, fhd):
    positions = np.random.uniform(var_min, var_max, (n_pop, n_var))
    fitness = np.full(n_pop, np.inf)
    best_position = np.zeros(n_var)
    best_cost = np.inf
    convergence = np.zeros(max_it)
    delta = np.random.rand(n_pop) * 2 * np.pi
    sigma1 = [np.random.rand(1, n_var) for _ in range(n_pop)]

    for i in range(n_pop):
        fitness[i] = fhd(positions[i, :])
        if fitness[i] <= best_cost:
            best_position = positions[i, :]
            best_cost = fitness[i]

    it = 0
    pos1 = np.zeros(n_var)
    while it <= max_it:
        best_cost = np.min(fitness)
        worst_cost = np.max(fitness)

        for i in range(n_pop):
            for jj in range(1, 6):
                R = fitness[i]
                C = (R / 2) * np.sin(delta[i])

                newsol = np.copy(positions[i, :])
                newsol2 = np.copy(positions[i, :])

                if jj == 1:
                    newsol += ((R ** 2 + (1 / (2 * np.pi * n_var * R * C) ** 2)) ** -0.5) * np.sin(
                        2 * np.pi * n_var * it) * np.sin((2 * np.pi * n_var * it) + delta[i]) * positions[i, :]
                else:
                    newsol += ((R ** 2 + (1 / (2 * np.pi * n_var * R * C) ** 2)) ** -0.5) * np.sin(
                        2 * np.pi * n_var * it) * np.sin((2 * np.pi * n_var * it) + delta[i]) * pos1

                perm = np.random.permutation(n_pop)
                a1, a2, a3 = perm[:3]

                aa1 = (fitness[a2] - fitness[a3]) / abs(fitness[a3] - fitness[a2])
                aa1 = 1.0 if fitness[a2] == fitness[a3] else aa1

                aa2 = (fitness[a1] - fitness[i]) / abs(fitness[a1] - fitness[i])
                aa2 = 1.0 if fitness[a1] == fitness[i] else aa2

                newsol += aa2 * sigma1[i] * (newsol - positions[a1, :]) + aa1 * sigma1[i] * (
                            positions[a3, :] - positions[a2, :])
                newsol2 = positions[a1, :] + sigma1[i] * (positions[a3, :] - positions[a2, :])

                for j in range(n_var):
                    pos1[j] = newsol2[j] if np.random.rand() / jj > np.random.rand() else newsol[j]

                pos1 = np.clip(pos1, var_min, var_max)
                newsol = pos1
                delta[i] = np.arctan(1 / (2 * np.pi * n_var * R * C))

                new_cost = fhd(newsol)
                if new_cost < fitness[i]:
                    positions[i, :] = newsol
                    fitness[i] = new_cost
                    if new_cost <= best_cost:
                        best_position = newsol
                        best_cost = new_cost

                sigma1[i] = np.random.rand(1, n_var)

        it += 1

    return best_cost, best_position, convergence
