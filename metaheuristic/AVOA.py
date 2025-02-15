
import numpy as np

def AVOA(pop_size, max_iter, lower_bound, upper_bound, variables_no, fobj):
    Best_vulture1_X = np.zeros(variables_no)
    Best_vulture1_F = np.inf
    Best_vulture2_X = np.zeros(variables_no)
    Best_vulture2_F = np.inf

    X = np.random.rand(pop_size, variables_no) * (upper_bound - lower_bound) + lower_bound

    p1, p2, p3 = 0.6, 0.4, 0.6
    alpha, beta, gamma = 0.8, 0.2, 2.5

    convergence_curve = np.zeros(max_iter)

    for It in range(max_iter):
        for i in range(pop_size):
            current_vulture_X = X[i, :]
            current_vulture_F = fobj(current_vulture_X)

            if current_vulture_F < Best_vulture1_F:
                Best_vulture1_F = current_vulture_F
                Best_vulture1_X = current_vulture_X
            elif current_vulture_F > Best_vulture1_F and current_vulture_F < Best_vulture2_F:
                Best_vulture2_F = current_vulture_F
                Best_vulture2_X = current_vulture_X

        a = np.random.uniform(-2, 2) * ((np.sin((np.pi / 2) * (It / max_iter))**gamma) + np.cos((np.pi / 2) * (It / max_iter)) - 1)
        P1 = (2 * np.random.rand() + 1) * (1 - (It / max_iter)) + a

        for i in range(pop_size):
            F = P1 * (2 * np.random.rand() - 1)

            if abs(F) >= 1:
                X[i, :] = Best_vulture1_X - abs((2 * np.random.rand()) * Best_vulture1_X - X[i, :]) * F
            else:
                A = Best_vulture1_X - ((Best_vulture1_X * X[i, :]) / (Best_vulture1_X - X[i, :]**2)) * F
                B = Best_vulture2_X - ((Best_vulture2_X * X[i, :]) / (Best_vulture2_X - X[i, :]**2)) * F
                X[i, :] = (A + B) / 2

        X = np.clip(X, lower_bound, upper_bound)

        convergence_curve[It] = Best_vulture1_F

    return Best_vulture1_F, Best_vulture1_X, convergence_curve
