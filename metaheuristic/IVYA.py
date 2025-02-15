import numpy as np


def ivy_algorithm(n, max_iterations, lb, ub, dim, fobj):
    position = np.random.rand(n, dim) * (ub - lb) + lb
    gv = position / (ub - lb)
    cost = np.array([fobj(position[i, :]) for i in range(n)])

    best_costs = np.zeros(max_iterations)
    convergence_curve = np.zeros(max_iterations)

    for it in range(max_iterations):
        best_cost = np.min(cost)
        worst_cost = np.max(cost)
        new_pop = []

        for i in range(n):
            ii = (i + 1) % n
            beta_1 = 1 + np.random.rand() / 2

            if cost[i] < beta_1 * cost[0]:
                new_position = position[i, :] + np.abs(np.random.randn(dim)) * (
                            position[ii, :] - position[i, :]) + np.random.randn(dim) * gv[i, :]
            else:
                new_position = position[0, :] * (np.random.rand() + np.random.randn(dim) * gv[i, :])

            gv[i, :] *= (np.random.rand() ** 2) * np.random.randn(dim)
            new_position = np.clip(new_position, lb, ub)
            new_gv = new_position / (ub - lb)
            new_cost = fobj(new_position)
            new_pop.append((new_position, new_cost, new_gv))

        for new_position, new_cost, new_gv in new_pop:
            position = np.vstack((position, new_position))
            cost = np.append(cost, new_cost)
            gv = np.vstack((gv, new_gv))

        sorted_indices = np.argsort(cost)
        position = position[sorted_indices, :]
        cost = cost[sorted_indices]

        if len(cost) > n:
            position = position[:n, :]
            cost = cost[:n]
            gv = gv[:n, :]

        best_costs[it] = cost[0]
        convergence_curve[it] = cost[0]

    destination_fitness = cost[0]
    destination_position = position[0, :]

    return destination_fitness, destination_position, convergence_curve
