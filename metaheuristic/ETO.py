
import numpy as np

def ETO(N, Max_Iter, LB, UB, Dim, Fobj):
    """
    Exponential-Trigonometric Optimization (ETO)

    Parameters:
        N (int): Population size.
        Max_Iter (int): Maximum iterations.
        LB (float): Lower bound of variables.
        UB (float): Upper bound of variables.
        Dim (int): Dimensionality of the problem.
        Fobj (function): Objective function.

    Returns:
        tuple: (Best fitness, Best solution, Convergence curve)
    """
    Destination_position = np.zeros(Dim)
    Destination_fitness = float('inf')
    Destination_position_second = np.zeros(Dim)
    Convergence_curve = np.zeros(Max_Iter)
    Position_sort = np.zeros((N, Dim))

    b, CE, T = 1.55, int(1 + Max_Iter / 1.55), int(1.2 + Max_Iter / 2.25)
    CEi, CEi_temp = 0, 0
    UB_2, LB_2 = UB, LB

    X = np.random.uniform(LB, UB, (N, Dim))
    Objective_values = np.array([Fobj(X[i, :]) for i in range(N)])

    for i in range(N):
        if Objective_values[i] < Destination_fitness:
            Destination_position = X[i, :].copy()
            Destination_fitness = Objective_values[i]

    Convergence_curve[0] = Destination_fitness
    t = 1

    while t < Max_Iter:
        for i in range(N):
            for j in range(Dim):
                d1 = 0.1 * np.exp(-0.01 * t) * np.cos(0.5 * Max_Iter * (1 - t / Max_Iter))
                d2 = -0.1 * np.exp(-0.01 * t) * np.cos(0.5 * Max_Iter * (1 - t / Max_Iter))

                CM = (np.sqrt(t / Max_Iter) ** np.tan(d1 / d2)) * np.random.rand() * 0.01

                if t == CEi:
                    UB_2 = Destination_position[j] + (1 - t / Max_Iter) * np.abs(np.random.rand() * Destination_position[j] - Destination_position_second[j]) * np.random.rand()
                    LB_2 = Destination_position[j] - (1 - t / Max_Iter) * np.abs(np.random.rand() * Destination_position[j] - Destination_position_second[j]) * np.random.rand()

                    UB_2 = min(UB_2, UB)
                    LB_2 = max(LB_2, LB)

                    X = np.random.uniform(LB_2, UB_2, (N, Dim))
                    CEi_temp, CEi = CEi, 0

                if t <= T:
                    q1, q3, q4, q5 = np.random.rand(), np.random.rand(), np.random.rand(), np.random.rand()

                    if CM > 1:
                        d1 = 0.1 * np.exp(-0.01 * t) * np.cos(0.5 * Max_Iter * q1)
                        d2 = -0.1 * np.exp(-0.01 * t) * np.cos(0.5 * Max_Iter * q1)
                        alpha_1 = q5 * 3 * (t / Max_Iter - 0.85) * np.exp(np.abs(d1 / d2) - 1)

                        if q1 <= 0.5:
                            X[i, j] = Destination_position[j] + q5 * alpha_1 * np.abs(Destination_position[j] - X[i, j])
                        else:
                            X[i, j] = Destination_position[j] - q5 * alpha_1 * np.abs(Destination_position[j] - X[i, j])
                    else:
                        d1 = 0.1 * np.exp(-0.01 * t) * np.cos(0.5 * Max_Iter * q3)
                        d2 = -0.1 * np.exp(-0.01 * t) * np.cos(0.5 * Max_Iter * q3)
                        alpha_3 = np.random.rand() * 3 * (t / Max_Iter - 0.85) * np.exp(np.abs(d1 / d2) - 1.3)

                        if q3 <= 0.5:
                            X[i, j] = Destination_position[j] + q4 * alpha_3 * np.abs(q5 * Destination_position[j] - X[i, j])
                        else:
                            X[i, j] = Destination_position[j] - q4 * alpha_3 * np.abs(q5 * Destination_position[j] - X[i, j])
                else:
                    q2, q6 = np.random.rand(), np.random.rand()
                    alpha_2 = q6 * np.exp(np.tanh(1.5 * (-t / Max_Iter - 0.75) - q6))

                    if CM < 1:
                        d1 = 0.1 * np.exp(-0.01 * t) * np.cos(0.5 * Max_Iter * q2)
                        d2 = -0.1 * np.exp(-0.01 * t) * np.cos(0.5 * Max_Iter * q2)
                        X[i, j] += np.exp(np.tan(np.abs(d1 / d2)) * np.abs(q6 * alpha_2 * Destination_position[j] - X[i, j]))
                    else:
                        if q2 <= 0.5:
                            X[i, j] += 3 * np.abs(q6 * alpha_2 * Destination_position[j] - X[i, j])
                        else:
                            X[i, j] -= 3 * np.abs(q6 * alpha_2 * Destination_position[j] - X[i, j])

        CEi = CEi_temp

        for i in range(N):
            X[i, :] = np.clip(X[i, :], LB_2, UB_2)
            Objective_values[i] = Fobj(X[i, :])

            if Objective_values[i] < Destination_fitness:
                Destination_position = X[i, :].copy()
                Destination_fitness = Objective_values[i]

        Convergence_curve[t] = Destination_fitness
        t += 1

    return Destination_fitness, Destination_position, Convergence_curve
