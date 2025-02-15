import numpy as np


def coefficient_vector(dim, iter, max_iter):
    a2 = -1 + iter * (-1 / max_iter)
    u = np.random.randn(dim)
    v = np.random.randn(dim)

    cofi = np.zeros((4, dim))
    cofi[0, :] = np.random.rand(dim)
    cofi[1, :] = (a2 + 1) + np.random.rand(dim)
    cofi[2, :] = a2 * np.random.randn(dim)
    cofi[3, :] = u * (v ** 2) * np.cos(np.random.rand() * 2 * u)

    return cofi


def solution_imp(x, best_x, lb, ub, n, cofi, m, a, d, i):
    new_x = np.zeros((4, x.shape[1]))
    new_x[0, :] = np.random.rand(x.shape[1]) * (ub - lb) + lb
    new_x[1, :] = best_x - np.abs((np.random.randint(1, 3) * m - np.random.randint(1, 3) * x[i, :]) * a) * cofi[
                                                                                                           np.random.randint(
                                                                                                               4), :]
    new_x[2, :] = (m + cofi[np.random.randint(4), :]) + (
                np.random.randint(1, 3) * best_x - np.random.randint(1, 3) * x[np.random.randint(n), :]) * cofi[
                                                                                                           np.random.randint(
                                                                                                               4), :]
    new_x[3, :] = (x[i, :] - d) + (np.random.randint(1, 3) * best_x - np.random.randint(1, 3) * m) * cofi[
                                                                                                     np.random.randint(
                                                                                                         4), :]

    return new_x


def mountain_gazelle_optimizer(n, max_iter, lb, ub, dim, fobj):
    x = np.random.uniform(lb, ub, (n, dim))
    best_x = None
    best_fitness = np.inf
    sol_cost = np.zeros(n)

    for i in range(n):
        sol_cost[i] = fobj(x[i, :])
        if sol_cost[i] <= best_fitness:
            best_fitness = sol_cost[i]
            best_x = x[i, :]

    cnvg = np.zeros(max_iter)

    for iter in range(max_iter):
        for i in range(n):
            random_solution = np.random.choice(n, size=int(n / 3), replace=False)
            m = x[np.random.choice(range(int(n / 3), n))] * np.floor(np.random.rand()) + np.mean(x[random_solution, :],
                                                                                                 axis=0) * np.ceil(
                np.random.rand())

            cofi = coefficient_vector(dim, iter, max_iter)
            a = np.random.randn(dim) * np.exp(2 - iter * (2 / max_iter))
            d = (np.abs(x[i, :]) + np.abs(best_x)) * (2 * np.random.rand() - 1)

            new_x = solution_imp(x, best_x, lb, ub, n, cofi, m, a, d, i)
            new_x = np.clip(new_x, lb, ub)

            sol_cost_new = np.array([fobj(new_x[j, :]) for j in range(new_x.shape[0])])

            x = np.vstack((x, new_x))
            sol_cost = np.concatenate((sol_cost, sol_cost_new))

            best_idx = np.argmin(sol_cost)
            best_x = x[best_idx, :]

        sort_order = np.argsort(sol_cost)
        sol_cost = sol_cost[sort_order]
        x = x[sort_order, :]
        best_fitness = sol_cost[0]
        best_x = x[0, :]
        x = x[:n, :]
        sol_cost = sol_cost[:n]
        cnvg[iter] = best_fitness

    return best_fitness, best_x, cnvg
