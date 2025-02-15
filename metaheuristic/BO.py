
import numpy as np

def BO(N, max_it, Var_min, Var_max, d, CostFunction):
    best_cost = np.inf
    bonobo = np.random.rand(N, d) * (Var_max - Var_min) + Var_min
    cost = np.array([CostFunction(bonobo[i, :]) for i in range(N)])
    
    best_idx = np.argmin(cost)
    best_solution = bonobo[best_idx, :].copy()
    
    convergence = np.zeros(max_it)

    for it in range(max_it):
        for i in range(N):
            new_bonobo = bonobo[i, :] + np.random.rand(d) * (best_solution - bonobo[i, :])
            new_bonobo = np.clip(new_bonobo, Var_min, Var_max)
            new_cost = CostFunction(new_bonobo)

            if new_cost < cost[i]:
                bonobo[i, :] = new_bonobo
                cost[i] = new_cost

            if new_cost < best_cost:
                best_cost = new_cost
                best_solution = new_bonobo

        convergence[it] = best_cost

    return best_cost, best_solution, convergence
