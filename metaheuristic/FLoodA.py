
import numpy as np

def FLoodA(nPop, MaxIt, lb, ub, nVar, fobj):
    """
    Flood Algorithm (FLoodA)

    Parameters:
        nPop (int): Population size.
        MaxIt (int): Maximum iterations.
        lb (float): Lower bound of variables.
        ub (float): Upper bound of variables.
        nVar (int): Dimensionality of the problem.
        fobj (function): Objective function.

    Returns:
        tuple: (Best fitness, Best solution, Convergence curve)
    """
    VarMin, VarMax = lb, ub
    NFEs = 0  
    Ne = 5  
    Convergence_curve = []  

    Position = np.random.uniform(VarMin, VarMax, (nPop, nVar))  
    Cost = np.array([fobj(Position[i, :]) for i in range(nPop)])

    BestSolCost = np.min(Cost)
    BestSolPosition = Position[np.argmin(Cost), :].copy()

    it = 0

    while NFEs < nPop * MaxIt:
        it += 1
        PK = ((((MaxIt * (it**2) + 1)**0.5 + (1 / ((MaxIt / 4) * it)) * np.log(((MaxIt * (it**2) + 1)**0.5 + (MaxIt / 4) * it))))**(-2/3)) * (1.2 / it)

        for i in range(nPop):
            sorted_costs = np.sort(Cost)
            Pe_i = ((Cost[i] - sorted_costs[0]) / (sorted_costs[-1] - sorted_costs[0]))**2
            A = [x for x in range(nPop) if x != i]
            a = np.random.choice(A)

            if np.random.rand() > (np.random.rand() + Pe_i):
                Val = ((PK**np.random.randn()) / it) * (np.random.rand(nVar) * (VarMax - VarMin) + VarMin)
                new_position = Position[i, :] + Val
            else:
                new_position = BestSolPosition + np.random.rand(nVar) * (Position[a, :] - Position[i, :])

            new_position = np.clip(new_position, VarMin, VarMax)
            new_cost = fobj(new_position)
            NFEs += 1

            if new_cost <= Cost[i]:
                Position[i, :] = new_position
                Cost[i] = new_cost

            if new_cost <= BestSolCost:
                BestSolCost = new_cost
                BestSolPosition = new_position

        Pt = np.abs(np.sin(np.random.rand() / it))
        if np.random.rand() < Pt:
            sorted_indices = np.argsort(Cost)
            Position = Position[sorted_indices[:nPop - Ne], :]
            Cost = Cost[sorted_indices[:nPop - Ne]]
            
            for _ in range(Ne):
                new_position = BestSolPosition + np.random.rand() * (np.random.rand(nVar) * (VarMax - VarMin) + VarMin)
                new_cost = fobj(new_position)
                NFEs += 1

                Position = np.vstack((Position, new_position))
                Cost = np.append(Cost, new_cost)

                if new_cost <= BestSolCost:
                    BestSolCost = new_cost
                    BestSolPosition = new_position

        Convergence_curve.append(BestSolCost)

    return BestSolCost, BestSolPosition, Convergence_curve
