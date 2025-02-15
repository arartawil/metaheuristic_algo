import numpy as np


def hgso(n_gases, n_iter, lower_bound, upper_bound, dim, objfunc):
    n_types = 5

    l1, l2, l3 = 5e-3, 100, 1e-2
    alpha, beta = 1, 1
    M1, M2 = 0.1, 0.2

    K = l1 * np.random.rand(n_types)
    P = l2 * np.random.rand(n_gases)
    C = l3 * np.random.rand(n_types)

    X = lower_bound + np.random.rand(n_gases, dim) * (upper_bound - lower_bound)
    Group = create_groups(n_gases, n_types, X)

    best_fit = np.zeros(n_types)
    best_pos = [None] * n_types

    for i in range(n_types):
        Group[i], best_fit[i], best_pos[i] = evaluate(objfunc, n_types, n_gases, Group[i], None, 1)

    Gbest = np.min(best_fit)
    gbest_idx = np.argmin(best_fit)
    Xbest = best_pos[gbest_idx]

    Gbest_iter = np.zeros(n_iter)

    for iteration in range(n_iter):
        S = update_variables(iteration, n_iter, K, P, C, n_types, n_gases)
        Group_new = update_positions(Group, best_pos, Xbest, S, n_gases, n_types, Gbest, alpha, beta, dim)
        Group_new = check_positions(dim, Group_new, n_gases, n_types, lower_bound, upper_bound)

        for i in range(n_types):
            Group[i], best_fit[i], best_pos[i] = evaluate(objfunc, n_types, n_gases, Group[i], Group_new[i], 0)
            Group[i] = worst_agents(Group[i], M1, M2, dim, upper_bound, lower_bound, n_gases, n_types)

        Ybest = np.min(best_fit)
        Ybest_idx = np.argmin(best_fit)
        Gbest_iter[iteration] = Ybest

        if Ybest < Gbest:
            Gbest = Ybest
            Xbest = best_pos[Ybest_idx]

    return Gbest, Xbest, Gbest_iter


def create_groups(n_gases, n_types, X):
    N = n_gases // n_types
    Group = [{} for _ in range(n_types)]

    idx = 0
    for j in range(n_types):
        Group[j]['Position'] = X[idx:idx + N, :]
        idx = (j + 1) * N

    return Group


def evaluate(objfunc, n_types, n_gases, X, Xnew, init_flag):
    N = n_gases // n_types

    if 'fitness' not in X:
        X['fitness'] = np.zeros(N)

    if init_flag == 1:
        for j in range(N):
            X['fitness'][j] = objfunc(X['Position'][j, :])
    else:
        for j in range(N):
            temp_fit = objfunc(Xnew['Position'][j, :])
            if temp_fit < X['fitness'][j]:
                X['fitness'][j] = temp_fit
                X['Position'][j, :] = Xnew['Position'][j, :]

    best_idx = np.argmin(X['fitness'])
    return X, X['fitness'][best_idx], X['Position'][best_idx, :]


def update_variables(iteration, n_iter, K, P, C, n_types, n_gases):
    T = np.exp(-iteration / n_iter)
    T0 = 298.15
    N = n_gases // n_types
    S = np.zeros((n_gases, P.shape[0]))

    idx = 0
    for j in range(n_types):
        K[j] = K[j] * np.exp(-C[j] * (1 / T - 1 / T0))
        S[idx:idx + N, :] = P[idx:idx + N, :] * K[j]
        idx = (j + 1) * N

    return S


def update_positions(Group, best_pos, Xbest, S, n_gases, n_types, Gbest, alpha, beta, dim):
    vec_flag = [1, -1]

    for i in range(n_types):
        for j in range(n_gases // n_types):
            gamma = beta * np.exp(-(Gbest + 0.05) / (Group[i]['fitness'][j] + 0.05))
            var_flag = np.random.choice(vec_flag)

            for k in range(dim):
                Group[i]['Position'][j, k] += var_flag * np.random.rand() * gamma * (
                            best_pos[i][k] - Group[i]['Position'][j, k]) + np.random.rand() * alpha * var_flag * (
                                                          S[i] * Xbest[k] - Group[i]['Position'][j, k])

    return Group


def check_positions(dim, Group, n_gases, n_types, lower_bound, upper_bound):
    Lb, Ub = np.ones(dim) * lower_bound, np.ones(dim) * upper_bound

    for j in range(n_types):
        for i in range(n_gases // n_types):
            pos = Group[j]['Position'][i, :]
            pos = np.clip(pos, Lb, Ub)
            Group[j]['Position'][i, :] = pos

    return Group


def worst_agents(X, M1, M2, dim, G_max, G_min, n_gases, n_types):
    X_index = np.argsort(X['fitness'])[::-1]
    M1N, M2N = M1 * n_gases / n_types, M2 * n_gases / n_types
    Nw = int(round((M2N - M1N) * np.random.rand() + M1N))

    for k in range(Nw):
        idx = X_index[k]
        X['Position'][idx, :] = G_min + np.random.rand(dim) * (G_max - G_min)

    return X
