import numpy as np


def moss_growth_optimization(search_agents_no, max_it, lb, ub, dim, fobj):
    best_cost = np.inf
    best_m = np.zeros(dim)
    m = np.random.uniform(lb, ub, (search_agents_no, dim))
    costs = np.array([fobj(m[i, :]) for i in range(search_agents_no)])
    best_idx = np.argmin(costs)
    best_m = m[best_idx, :]
    best_cost = costs[best_idx]

    convergence_curve = []
    w = 2
    rec_num = 10
    divide_num = max(dim // 4, 1)
    d1 = 0.2

    new_m = np.zeros((search_agents_no, dim))
    new_m_cost = np.zeros(search_agents_no)
    r_m = np.zeros((search_agents_no, dim, rec_num))
    r_m_cos = np.zeros((search_agents_no, rec_num))

    for _ in range(max_it):
        cal_positions = m.copy()
        div_num = np.random.permutation(dim)

        for j in range(divide_num):
            th = best_m[div_num[j]]
            index = cal_positions[:, div_num[j]] > th
            if np.sum(index) < search_agents_no / 2:
                index = ~index
            cal_positions = cal_positions[index, :]

        d = best_m - np.mean(cal_positions, axis=0)

        beta = len(cal_positions) / search_agents_no
        gama = 1 / np.sqrt(1 - beta ** 2)
        step = w * (np.random.rand(len(d)) - 0.5) * (1 - _ / max_it)
        step2 = 0.1 * w * (len(d) - 0.5) * (1 - _ / max_it) * (1 + 0.5 * (1 + np.tanh(beta / gama)) * (1 - _ / max_it))
        step3 = 0.1 * (np.random.rand() - 0.5) * (1 - _ / max_it)
        rand_vals = np.random.rand(len(d))
        result = 1 / (1 + (0.5 - 10 * rand_vals))
        act = result >= 0.5

        for i in range(search_agents_no):
            new_m[i, :] = m[i, :]
            if np.random.rand() > d1:
                new_m[i, :] += step * d
            else:
                new_m[i, :] += step2 * d

            if np.random.rand() < 0.8:
                if np.random.rand() > 0.5:
                    new_m[i, div_num[0]] = best_m[div_num[0]] + step3 * d[div_num[0]]
                else:
                    new_m[i, :] = (1 - act) * new_m[i, :] + act * best_m

            new_m[i, :] = np.clip(new_m[i, :], lb, ub)
            new_m_cost[i] = fobj(new_m[i, :])

            if new_m_cost[i] < best_cost:
                best_m = new_m[i, :]
                best_cost = new_m_cost[i]

        convergence_curve.append(best_cost)

    return best_cost, best_m, convergence_curve