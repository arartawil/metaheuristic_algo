import numpy as np

def HGS(N, Max_iter, lb, ub, dim, fobj):
    bestPositions = np.zeros(dim)
    tempPosition = np.zeros((N, dim))
    Destination_fitness = float('inf')
    Worstest_fitness = -float('inf')
    AllFitness = np.full(N, float('inf'))
    VC1 = np.ones(N)

    weight3 = np.ones((N, dim))
    weight4 = np.ones((N, dim))

    X = initialization(N, dim, ub, lb)
    Convergence_curve = np.zeros(Max_iter)
    it = 1
    hungry = np.zeros(N)
    count = 0

    # Main loop
    while it <= Max_iter:
        VC2 = 0.03
        sumHungry = 0

        for i in range(N):
            X[i, :] = np.clip(X[i, :], lb, ub)
            AllFitness[i] = fobj(X[i, :])

        IndexSorted = np.argsort(AllFitness)  # Get the indices that would sort the array
        AllFitnessSorted = AllFitness[IndexSorted]
        bestFitness = AllFitnessSorted[0]
        worstFitness = AllFitnessSorted[-1]

        if bestFitness < Destination_fitness:
            bestPositions = X[IndexSorted[0], :]
            Destination_fitness = bestFitness
            count = 0

        if worstFitness > Worstest_fitness:
            Worstest_fitness = worstFitness

        for i in range(N):
            VC1[i] = 1 / np.cosh(abs(AllFitness[i] - Destination_fitness))

            if Destination_fitness == AllFitness[i]:
                hungry[i] = 0
                count += 1
                tempPosition[count, :] = X[i, :]
            else:
                temprand = np.random.rand()
                c = (AllFitness[i] - Destination_fitness) / (Worstest_fitness - Destination_fitness) * temprand * 2 * (ub - lb)
                b = 100 * (1 + temprand) if c < 100 else c
                hungry[i] += np.max(b)
                sumHungry += hungry[i]

        for i in range(N):
            for j in range(1, dim):
                weight3[i, j] = (1 - np.exp(-abs(hungry[i] - sumHungry))) * np.random.rand() * 2
                if np.random.rand() < VC2:
                    weight4[i, j] = hungry[i] * N / sumHungry * np.random.rand()
                else:
                    weight4[i, j] = 1

        shrink = 2 * (1 - it / Max_iter)
        for i in range(N):
            if np.random.rand() < VC2:
                X[i, :] *= 1 + np.random.randn()
            else:
                A = np.random.randint(1, count + 1)
                for j in range(dim):
                    r = np.random.rand()
                    vb = 2 * shrink * r - shrink
                    if r > VC1[i]:
                        X[i, j] = weight4[i, j] * tempPosition[A, j] + vb * weight3[i, j] * abs(tempPosition[A, j] - X[i, j])
                    else:
                        X[i, j] = weight4[i, j] * tempPosition[A, j] - vb * weight3[i, j] * abs(tempPosition[A, j] - X[i, j])

        Convergence_curve[it - 1] = Destination_fitness
        it += 1

    return Destination_fitness, bestPositions, Convergence_curve

def initialization(N, dim, ub, lb):
    if isinstance(ub, (int, float)):
        ub = np.full(dim, ub)
    if isinstance(lb, (int, float)):
        lb = np.full(dim, lb)
    return np.random.uniform(lb, ub, (N, dim))

