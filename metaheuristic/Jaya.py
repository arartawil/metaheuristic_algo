import numpy as np


def jaya(pop, max_gen, mini, maxi, var, objective):
    if isinstance(mini, (int, float)):
        mini = np.full(var, mini)
    if isinstance(maxi, (int, float)):
        maxi = np.full(var, maxi)

    x = np.random.rand(pop, var) * (maxi - mini) + mini
    f = np.array([objective(x[i, :]) for i in range(pop)])
    convergence = np.zeros(max_gen)
    fopt = np.zeros(max_gen)
    xopt = np.zeros((max_gen, var))

    for gen in range(max_gen):
        best_idx = np.argmin(f)
        worst_idx = np.argmax(f)
        best = x[best_idx, :]
        worst = x[worst_idx, :]
        x_new = np.zeros_like(x)

        for i in range(pop):
            x_new[i, :] = x[i, :] + np.random.rand() * (best - np.abs(x[i, :])) - (worst - np.abs(x[i, :]))

        x_new = np.clip(x_new, mini, maxi)
        f_new = np.array([objective(x_new[i, :]) for i in range(pop)])

        for i in range(pop):
            if f_new[i] < f[i]:
                x[i, :] = x_new[i, :]
                f[i] = f_new[i]

        fopt[gen] = np.min(f)
        xopt[gen, :] = x[np.argmin(f), :]
        convergence[gen] = fopt[gen]

    best_value = np.min(fopt)
    best_position = xopt[np.argmin(fopt), :]

    return best_value, best_position, convergence
