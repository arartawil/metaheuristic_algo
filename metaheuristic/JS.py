import numpy as np


def jellyfish_optimization(n_pop, max_it, lb, ub, nd, cost_function):
    if isinstance(lb, (int, float)):
        var_min = np.full(nd, lb)
        var_max = np.full(nd, ub)
    else:
        var_min, var_max = lb, ub

    pop = np.random.rand(n_pop, nd) * (var_max - var_min) + var_min
    pop_cost = np.array([cost_function(pop[i, :]) for i in range(n_pop)])

    best_solution = np.zeros(nd)
    best_cost = float('inf')
    convergence_curve = np.zeros(max_it)

    for it in range(max_it):
        mean_val = np.mean(pop, axis=0)
        sorted_indices = np.argsort(pop_cost)
        best_solution = pop[sorted_indices[0], :]
        best_cost = pop_cost[sorted_indices[0]]

        for i in range(n_pop):
            ar = (1 - it / max_it) * (2 * np.random.rand() - 1)

            if abs(ar) >= 0.5:
                new_sol = pop[i, :] + np.random.rand(nd) * (best_solution - 3 * np.random.rand() * mean_val)
            else:
                if np.random.rand() <= (1 - ar):
                    j = np.random.randint(n_pop)
                    while j == i:
                        j = np.random.randint(n_pop)
                    step = pop[i, :] - pop[j, :]
                    if pop_cost[j] < pop_cost[i]:
                        step = -step
                    new_sol = pop[i, :] + np.random.rand(nd) * step
                else:
                    new_sol = pop[i, :] + 0.1 * (var_max - var_min) * np.random.rand()

            new_sol = np.clip(new_sol, var_min, var_max)
            new_sol_cost = cost_function(new_sol)

            if new_sol_cost < pop_cost[i]:
                pop[i, :] = new_sol
                pop_cost[i] = new_sol_cost
                if new_sol_cost < best_cost:
                    best_cost = new_sol_cost
                    best_solution = new_sol

        convergence_curve[it] = best_cost

    return best_cost, best_solution, convergence_curve
