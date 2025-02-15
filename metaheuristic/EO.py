
import numpy as np

def EO(Particles_no, Max_iter, lb, ub, dim, fobj):
    """
    Equilibrium Optimizer (EO)

    Parameters:
        Particles_no (int): Population size.
        Max_iter (int): Maximum iterations.
        lb (float): Lower bound of variables.
        ub (float): Upper bound of variables.
        dim (int): Dimensionality of the problem.
        fobj (function): Objective function.

    Returns:
        tuple: (Best fitness, Best solution, Convergence curve)
    """
    Convergence_curve = np.zeros(Max_iter)

    Ceq1, Ceq2, Ceq3, Ceq4 = np.zeros(dim), np.zeros(dim), np.zeros(dim), np.zeros(dim)
    Ceq1_fit, Ceq2_fit, Ceq3_fit, Ceq4_fit = float('inf'), float('inf'), float('inf'), float('inf')

    C = np.random.uniform(lb, ub, (Particles_no, dim))
    C_old = np.zeros_like(C)
    Iter, V = 0, 1

    a1, a2, GP = 2, 1, 0.5

    while Iter < Max_iter:
        fitness = np.array([fobj(C[i, :]) for i in range(Particles_no)])
        fit_old = fitness.copy()

        for i in range(Particles_no):
            if fitness[i] < Ceq1_fit:
                Ceq1_fit, Ceq1 = fitness[i], C[i, :].copy()
            elif Ceq1_fit < fitness[i] < Ceq2_fit:
                Ceq2_fit, Ceq2 = fitness[i], C[i, :].copy()
            elif Ceq2_fit < fitness[i] < Ceq3_fit:
                Ceq3_fit, Ceq3 = fitness[i], C[i, :].copy()
            elif Ceq3_fit < fitness[i] < Ceq4_fit:
                Ceq4_fit, Ceq4 = fitness[i], C[i, :].copy()

        if Iter == 0:
            C_old = C.copy()

        for i in range(Particles_no):
            if fit_old[i] < fitness[i]:
                fitness[i] = fit_old[i]
                C[i, :] = C_old[i, :]

        C_old = C.copy()
        fit_old = fitness.copy()

        Ceq_ave = (Ceq1 + Ceq2 + Ceq3 + Ceq4) / 4
        C_pool = np.vstack((Ceq1, Ceq2, Ceq3, Ceq4, Ceq_ave))

        t = (1 - Iter / Max_iter) ** (a2 * Iter / Max_iter)

        for i in range(Particles_no):
            lambda_vec = np.random.rand(dim)
            r = np.random.rand(dim)

            Ceq = C_pool[np.random.randint(C_pool.shape[0]), :]
            F = a1 * np.sign(r - 0.5) * (np.exp(-lambda_vec * t) - 1)
            r1, r2 = np.random.rand(), np.random.rand()
            GCP = 0.5 * r1 * np.ones(dim) * (r2 >= GP)
            G0 = GCP * (Ceq - lambda_vec * C[i, :])
            G = G0 * F
            C[i, :] = Ceq + (C[i, :] - Ceq) * F + (G / lambda_vec * V) * (1 - F)

        Iter += 1
        Convergence_curve[Iter - 1] = Ceq1_fit

    return Ceq1_fit, Ceq1, Convergence_curve
