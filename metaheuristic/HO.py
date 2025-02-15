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


def hippopotamus_oa(search_agents, max_iterations, lower_bound, upper_bound, dimension, fitness):
    lower_bound = np.full(dimension, lower_bound)
    upper_bound = np.full(dimension, upper_bound)

    X = np.random.uniform(lower_bound, upper_bound, (search_agents, dimension))
    fit = np.array([fitness(X[i, :]) for i in range(search_agents)])
    Xbest, fbest = None, float('inf')

    best_so_far = np.zeros(max_iterations)
    for t in range(max_iterations):
        best, location = np.min(fit), np.argmin(fit)
        if t == 0 or best < fbest:
            Xbest = X[location, :].copy()
            fbest = best

        for i in range(search_agents // 2):
            dominant_hippo = Xbest
            I1, I2 = np.random.choice([1, 2]), np.random.choice([1, 2])
            rand_group = np.random.choice(search_agents, np.random.randint(1, search_agents), replace=False)
            mean_group = np.mean(X[rand_group, :], axis=0) if len(rand_group) > 1 else X[rand_group[0], :]

            alfa = [
                I2 * np.random.rand(dimension),
                2 * np.random.rand(dimension) - 1,
                np.random.rand(dimension),
                I1 * np.random.rand(dimension),
                np.random.rand()
            ]

            A, B = np.random.choice(alfa), np.random.choice(alfa)
            X_P1 = X[i, :] + np.random.rand() * (dominant_hippo - I1 * X[i, :])
            T = np.exp(-t / max_iterations)

            if T > 0.6:
                X_P2 = X[i, :] + A * (dominant_hippo - I2 * mean_group)
            else:
                if np.random.rand() > 0.5:
                    X_P2 = X[i, :] + B * (mean_group - dominant_hippo)
                else:
                    X_P2 = np.random.uniform(lower_bound, upper_bound, dimension)
            X_P2 = np.clip(X_P2, lower_bound, upper_bound)

            F_P1, F_P2 = fitness(X_P1), fitness(X_P2)
            if F_P1 < fit[i]:
                X[i, :] = X_P1
                fit[i] = F_P1
            if F_P2 < fit[i]:
                X[i, :] = X_P2
                fit[i] = F_P2

        best_so_far[t] = fbest

    return fbest, Xbest, best_so_far
