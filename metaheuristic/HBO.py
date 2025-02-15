import numpy as np

def colleaguesLimitsGenerator(degree, searchAgents):
    colleaguesLimits = np.zeros((searchAgents, 2))
    for c in range(searchAgents, 0, -1):
        hi = np.ceil(np.log10(c * degree - c + 1) / np.log10(degree)) - 1
        lowerLim = (degree * degree ** (hi - 1) - 1) / (degree - 1) + 1
        upperLim = (degree * degree ** hi - 1) / (degree - 1)
        colleaguesLimits[c - 1, 0] = lowerLim
        colleaguesLimits[c - 1, 1] = upperLim

    return colleaguesLimits

def HBO(searchAgents, Max_iter, lb, ub, dim, fobj):
    cycles = int(np.floor(Max_iter / 25))
    degree = 3

    treeHeight = np.ceil(np.log10(searchAgents * degree - searchAgents + 1) / np.log10(degree))
    fevals = 0

    Leader_pos = np.zeros(dim)
    Leader_score = float('inf')

    Solutions = initialization(searchAgents, dim, ub, lb)

    fitnessHeap = np.full((searchAgents, 2), float('inf'))

    for c in range(searchAgents):
        fitness = fobj(Solutions[c, :])
        fevals += 1
        fitnessHeap[c, 0] = fitness
        fitnessHeap[c, 1] = c

        t = c
        while t > 0:
            parentInd = int(np.floor((t + 1) / degree))
            if fitnessHeap[t, 0] >= fitnessHeap[parentInd, 0]:
                break
            else:
                fitnessHeap[t, :], fitnessHeap[parentInd, :] = fitnessHeap[parentInd, :], fitnessHeap[t, :]
            t = parentInd

        if fitness <= Leader_score:
            Leader_score = fitness
            Leader_pos = Solutions[c, :]

    colleaguesLimits = colleaguesLimitsGenerator(degree, searchAgents)
    Convergence_curve = np.zeros(Max_iter)
    itPerCycle = Max_iter / cycles
    qtrCycle = itPerCycle / 4

    for it in range(Max_iter):
        gamma = abs(2 - (it % itPerCycle) / qtrCycle)

        for c in range(searchAgents, 1, -1):
            if c == 0:
                continue
            else:
                parentInd = (c + 1) // degree
                curSol = Solutions[int(fitnessHeap[c, 1]), :]
                parentSol = Solutions[int(fitnessHeap[parentInd, 1]), :]

                if colleaguesLimits[c, 1] > searchAgents:
                    colleaguesLimits[c, 1] = searchAgents

                colleagueInd = c
                while colleagueInd == c:
                    colleagueInd = int(np.random.randint(1, colleaguesLimits[c, 1] + 1))
                colleagueSol = Solutions[int(fitnessHeap[colleagueInd, 1]), :]

                for j in range(dim):
                    p1 = 1 - it / Max_iter
                    p2 = p1 + (1 - p1) / 2
                    r = np.random.rand()
                    rn = 2 * np.random.rand() - 1

                    if r < p1:
                        continue
                    elif r < p2:
                        D = abs(parentSol[j] - curSol[j])
                        curSol[j] = parentSol[j] + rn * gamma * D
                    else:
                        if fitnessHeap[colleagueInd, 0] < fitnessHeap[c, 0]:
                            D = abs(colleagueSol[j] - curSol[j])
                            curSol[j] = colleagueSol[j] + rn * gamma * D
                        else:
                            D = abs(colleagueSol[j] - curSol[j])
                            curSol[j] = curSol[j] + rn * gamma * D

                Flag4ub = curSol > ub
                Flag4lb = curSol < lb
                curSol = np.clip(curSol, lb, ub)

                newFitness = fobj(curSol)
                fevals += 1
                if newFitness < fitnessHeap[c, 0]:
                    fitnessHeap[c, 0] = newFitness
                    Solutions[int(fitnessHeap[c, 1]), :] = curSol
                if newFitness < Leader_score:
                    Leader_score = newFitness
                    Leader_pos = curSol

                t = c
                while t > 0:
                    parentInd = int(np.floor((t + 1) / degree))
                    if fitnessHeap[t, 0] >= fitnessHeap[parentInd, 0]:
                        break
                    else:
                        fitnessHeap[t, :], fitnessHeap[parentInd, :] = fitnessHeap[parentInd, :], fitnessHeap[t, :]
                    t = parentInd

        Convergence_curve[it] = Leader_score

    return Leader_score, Leader_pos, Convergence_curve

def initialization(pop, dim, ub, lb):
    if isinstance(ub, (int, float)):
        ub = np.full(dim, ub)
    if isinstance(lb, (int, float)):
        lb = np.full(dim, lb)
    return np.random.uniform(lb, ub, (pop, dim))