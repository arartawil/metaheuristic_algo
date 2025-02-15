
import numpy as np

class Mongoose:
    def __init__(self, position, cost):
        self.position = position
        self.cost = cost

def RouletteWheelSelection(P):
    r = np.random.rand()
    C = np.cumsum(P)
    return np.where(r <= C)[0][0]

def DMOA(nPop, MaxIt, VarMin, VarMax, nVar, F_obj):
    """
    Dwarf Mongoose Optimization Algorithm (DMOA)

    Parameters:
        nPop (int): Population size.
        MaxIt (int): Maximum iterations.
        VarMin (float): Lower bound of variables.
        VarMax (float): Upper bound of variables.
        nVar (int): Dimensionality of the problem.
        F_obj (function): Objective function.

    Returns:
        tuple: (Best fitness, Best solution, Convergence curve)
    """
    VarSize = nVar

    nBabysitter = 3
    nAlphaGroup = nPop - nBabysitter
    nScout = nAlphaGroup

    L = int(0.6 * nVar * nBabysitter)
    peep = 2.0

    pop = [Mongoose(np.random.uniform(VarMin, VarMax, VarSize), float('inf')) for _ in range(nAlphaGroup)]
    BestSol = Mongoose(np.zeros(VarSize), float('inf'))
    tau = float('inf')
    sm = np.full(nAlphaGroup, float('inf'))

    for i in range(nAlphaGroup):
        pop[i].cost = F_obj(pop[i].position)
        if pop[i].cost <= BestSol.cost:
            BestSol = pop[i]

    C = np.zeros(nAlphaGroup)
    CF = (1 - 1 / MaxIt) ** (2 * 1 / MaxIt)

    BestCost = np.zeros(MaxIt)

    for it in range(MaxIt):
        F = np.exp(-np.array([pop[i].cost for i in range(nAlphaGroup)]) / np.mean([pop[i].cost for i in range(nAlphaGroup)]))
        P = F / np.sum(F)

        for m in range(nAlphaGroup):
            i = RouletteWheelSelection(P)

            K = list(set(range(nAlphaGroup)) - {i})
            k = np.random.choice(K)

            phi = (peep / 2) * np.random.rand(VarSize) * (2 * np.random.rand(VarSize) - 1)
            newpop = Mongoose(pop[i].position + phi * (pop[i].position - pop[k].position), float('inf'))
            newpop.cost = F_obj(newpop.position)

            if newpop.cost <= pop[i].cost:
                pop[i] = newpop
            else:
                C[i] += 1

        for i in range(nScout):
            K = list(set(range(nAlphaGroup)) - {i})
            k = np.random.choice(K)

            phi = (peep / 2) * np.random.rand(VarSize) * (2 * np.random.rand(VarSize) - 1)
            newpop = Mongoose(pop[i].position + phi * (pop[i].position - pop[k].position), float('inf'))
            newpop.cost = F_obj(newpop.position)

            sm[i] = (newpop.cost - pop[i].cost) / max(newpop.cost, pop[i].cost)

            if newpop.cost <= pop[i].cost:
                pop[i] = newpop
            else:
                C[i] += 1

        for i in range(nBabysitter):
            if C[i] >= L:
                pop[i].position = np.random.uniform(VarMin, VarMax, VarSize)
                pop[i].cost = F_obj(pop[i].position)
                C[i] = 0

        for i in range(nAlphaGroup):
            if pop[i].cost <= BestSol.cost:
                BestSol = pop[i]

        newtau = np.mean(sm)
        for i in range(nScout):
            M = (pop[i].position * sm[i]) / pop[i].position
            newpop = Mongoose(pop[i].position - CF * np.random.rand(VarSize) * (pop[i].position - M) if newtau > tau else pop[i].position + CF * np.random.rand(VarSize) * (pop[i].position - M), float('inf'))
            tau = newtau

        for i in range(nAlphaGroup):
            if pop[i].cost <= BestSol.cost:
                BestSol = pop[i]

        BestCost[it] = BestSol.cost

    return BestSol.cost, BestSol.position, BestCost
