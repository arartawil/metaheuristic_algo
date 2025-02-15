import numpy as np
from scipy.special import gamma


def levy(d):
    beta = 1.5
    sigma = (gamma(1 + beta) * np.sin(np.pi * beta / 2) / (gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (
                1 / beta)
    u = np.random.randn(d) * sigma
    v = np.random.randn(d)
    step = u / np.abs(v) ** (1 / beta)
    return step


def horse_oa(n_horses, max_iter, var_min, var_max, n_var, cost_function):
    var_size = (1, n_var)
    vel_max = 0.1 * (var_max - var_min)
    vel_min = -vel_max

    horses = [{'Position': None, 'Cost': None, 'Velocity': None, 'Best': {'Position': None, 'Cost': float('inf')}} for _
              in range(n_horses)]
    global_best = {'Cost': float('inf')}

    for i in range(n_horses):
        horses[i]['Position'] = var_min + (var_max - var_min) * np.random.rand(n_var)
        horses[i]['Velocity'] = np.zeros(var_size)
        horses[i]['Cost'] = cost_function(horses[i]['Position'])

        horses[i]['Best']['Position'] = horses[i]['Position']
        horses[i]['Best']['Cost'] = horses[i]['Cost']

        if horses[i]['Best']['Cost'] < global_best['Cost']:
            global_best = horses[i]['Best']

    best_cost = np.zeros(max_iter)

    for it in range(max_iter):
        mean_position = np.mean([horse['Position'] for horse in horses], axis=0)
        bad_position = np.mean([horse['Position'] for horse in horses[int(0.9 * n_horses):]], axis=0)
        good_position = np.mean([horse['Position'] for horse in horses[:int(0.1 * n_horses)]], axis=0)

        for i in range(n_horses):
            if i < 0.1 * n_horses:
                horses[i]['Velocity'] = (
                            1.5 * np.random.rand(var_size) * (global_best['Position'] - horses[i]['Position']) -
                            0.5 * np.random.rand(var_size) * horses[i]['Position'] +
                            1.5 * (0.95 + 0.1 * np.random.rand()) * (
                                        horses[i]['Best']['Position'] - horses[i]['Position']))
            elif i < 0.3 * n_horses:
                horses[i]['Velocity'] = (0.2 * np.random.rand(var_size) * (mean_position - horses[i]['Position']) -
                                         0.2 * np.random.rand(var_size) * (bad_position - horses[i]['Position']) +
                                         0.9 * np.random.rand(var_size) * (
                                                     global_best['Position'] - horses[i]['Position']) +
                                         1.5 * (0.95 + 0.1 * np.random.rand()) * (
                                                     horses[i]['Best']['Position'] - horses[i]['Position']))
            else:
                horses[i]['Velocity'] = 1.5 * (0.95 + 0.1 * np.random.rand()) * (
                            horses[i]['Best']['Position'] - horses[i]['Position'])

            horses[i]['Velocity'] = np.clip(horses[i]['Velocity'], vel_min, vel_max)
            horses[i]['Position'] += horses[i]['Velocity']
            horses[i]['Position'] = np.clip(horses[i]['Position'], var_min, var_max)

            horses[i]['Cost'] = cost_function(horses[i]['Position'])

            if horses[i]['Cost'] < horses[i]['Best']['Cost']:
                horses[i]['Best']['Position'] = horses[i]['Position']
                horses[i]['Best']['Cost'] = horses[i]['Cost']

                if horses[i]['Best']['Cost'] < global_best['Cost']:
                    global_best = horses[i]['Best']

        best_cost[it] = global_best['Cost']

    return global_best['Cost'], global_best['Position'], best_cost
