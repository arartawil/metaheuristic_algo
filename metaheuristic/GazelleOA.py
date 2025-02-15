import numpy as np
from scipy.special import gamma

def levy_fun(n, m, beta=1.5):
    num = gamma(1 + beta) * np.sin(np.pi * beta / 2)
    den = gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2)
    sigma_u = (num / den) ** (1 / beta)

    u = np.random.normal(0, sigma_u, (n, m))
    v = np.random.normal(0, 1, (n, m))
    z = u / np.abs(v) ** (1 / beta)
    return z

def initialization(pop_size, dim, ub, lb):
    return lb + np.random.rand(pop_size, dim) * (ub - lb)

def GazelleOA(SearchAgents_no, Max_iter, lb, ub, dim, fobj):
    Top_gazelle_pos = np.zeros(dim)
    Top_gazelle_fit = float('inf')

    Convergence_curve = np.zeros(Max_iter)
    stepsize = np.zeros((SearchAgents_no, dim))
    fitness = np.full(SearchAgents_no, float('inf'))
    fit_old = np.full(SearchAgents_no, float('inf'))

    gazelle = initialization(SearchAgents_no, dim, ub, lb)
    Prey_old = np.zeros((SearchAgents_no, dim))

    Xmin = np.full((SearchAgents_no, dim), lb)
    Xmax = np.full((SearchAgents_no, dim), ub)

    Iter = 0
    PSRs = 0.34
    S = 0.88

    while Iter < Max_iter:
        for i in range(SearchAgents_no):
            gazelle[i, :] = np.clip(gazelle[i, :], lb, ub)
            fitness[i] = fobj(gazelle[i, :])

            if fitness[i] < Top_gazelle_fit:
                Top_gazelle_fit = fitness[i]
                Top_gazelle_pos = gazelle[i, :]

        if Iter == 0:
            fit_old = fitness.copy()
            Prey_old = gazelle.copy()

        Inx = fit_old < fitness
        gazelle = np.where(Inx[:, None], Prey_old, gazelle)
        fitness = np.where(Inx, fit_old, fitness)

        fit_old = fitness.copy()
        Prey_old = gazelle.copy()

        Elite = np.tile(Top_gazelle_pos, (SearchAgents_no, 1))
        CF = (1 - Iter / Max_iter) ** (2 * Iter / Max_iter)

        RL = 0.05 * levy_fun(SearchAgents_no, dim)
        RB = np.random.randn(SearchAgents_no, dim)

        for i in range(SearchAgents_no):
            for j in range(dim):
                R = np.random.rand()
                r = np.random.rand()
                mu = -1 if Iter % 2 == 0 else 1

                if r > 0.5:
                    stepsize[i, j] = RB[i, j] * (Elite[i, j] - RB[i, j] * gazelle[i, j])
                    gazelle[i, j] += np.random.rand() * R * stepsize[i, j]
                else:
                    if i > SearchAgents_no / 2:
                        stepsize[i, j] = RB[i, j] * (RL[i, j] * Elite[i, j] - gazelle[i, j])
                        gazelle[i, j] = Elite[i, j] + S * mu * CF * stepsize[i, j]
                    else:
                        stepsize[i, j] = RL[i, j] * (Elite[i, j] - RL[i, j] * gazelle[i, j])
                        gazelle[i, j] += S * mu * R * stepsize[i, j]

        for i in range(SearchAgents_no):
            gazelle[i, :] = np.clip(gazelle[i, :], lb, ub)
            fitness[i] = fobj(gazelle[i, :])

            if fitness[i] < Top_gazelle_fit:
                Top_gazelle_fit = fitness[i]
                Top_gazelle_pos = gazelle[i, :]

        Inx = fit_old < fitness
        gazelle = np.where(Inx[:, None], Prey_old, gazelle)
        fitness = np.where(Inx, fit_old, fitness)

        fit_old = fitness.copy()
        Prey_old = gazelle.copy()

        if np.random.rand() < PSRs:
            U = np.random.rand(SearchAgents_no, dim) < PSRs
            gazelle += CF * ((Xmin + np.random.rand(SearchAgents_no, dim) * (Xmax - Xmin)) * U)
        else:
            r = np.random.rand()
            Rs = SearchAgents_no
            perm_indices1 = np.random.permutation(Rs)
            perm_indices2 = np.random.permutation(Rs)
            stepsize = (PSRs * (1 - r) + r) * (gazelle[perm_indices1, :] - gazelle[perm_indices2, :])
            gazelle += stepsize

        Iter += 1
        Convergence_curve[Iter - 1] = Top_gazelle_fit

    return Top_gazelle_fit, Top_gazelle_pos, Convergence_curve
