
import numpy as np

def DDAO(Npop, MaxIt, L_limit, U_limit, Nvar, CostFunction):
    """
    Dynamic Differential Annealed Optimization (DDAO) Algorithm

    Parameters:
        Npop (int): Population size.
        MaxIt (int): Maximum iterations.
        L_limit (float): Lower bound of variables.
        U_limit (float): Upper bound of variables.
        Nvar (int): Dimensionality of the problem.
        CostFunction (function): Objective function.

    Returns:
        tuple: (Best fitness, Best solution, Convergence curve)
    """
    VarLength = Nvar
    MaxSubIt = 1000
    T0 = 2000.0
    alpha = 0.995

    pop = [{'Phase': np.random.uniform(L_limit, U_limit, VarLength), 'Cost': float('inf')} for _ in range(Npop)]
    BestSol = {'Phase': np.zeros(VarLength), 'Cost': float('inf')}

    for i in range(Npop):
        pop[i]['Cost'] = CostFunction(pop[i]['Phase'])
        if pop[i]['Cost'] <= BestSol['Cost']:
            BestSol = pop[i].copy()

    BestCost = np.zeros(MaxIt)
    T = T0

    for t in range(MaxIt):
        newpop = [{'Phase': np.random.uniform(L_limit, U_limit, VarLength), 'Cost': CostFunction(np.random.uniform(L_limit, U_limit, VarLength))} for _ in range(MaxSubIt)]

        newpop.sort(key=lambda x: x['Cost'])
        bnew = newpop[0]
        kk, bb = np.random.randint(0, Npop, size=2)

        Mnew = {'Phase': np.zeros(VarLength), 'Cost': float('inf')}
        if t % 2 == 1:
            Mnew['Phase'] = (pop[kk]['Phase'] - pop[bb]['Phase']) + bnew['Phase']
        else:
            Mnew['Phase'] = (pop[kk]['Phase'] - pop[bb]['Phase']) + bnew['Phase'] * np.random.rand()

        Mnew['Phase'] = np.clip(Mnew['Phase'], L_limit, U_limit)
        Mnew['Cost'] = CostFunction(Mnew['Phase'])

        for i in range(Npop):
            if Mnew['Cost'] <= pop[i]['Cost']:
                pop[i] = Mnew.copy()
            else:
                DELTA = Mnew['Cost'] - pop[i]['Cost']
                P = np.exp(-DELTA / T)
                if np.random.rand() <= P:
                    pop[-1] = Mnew.copy()

            if pop[i]['Cost'] <= BestSol['Cost']:
                BestSol = pop[i].copy()

        T *= alpha

    return BestSol['Cost'], BestSol['Phase'], BestCost
