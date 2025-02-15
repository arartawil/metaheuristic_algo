
import numpy as np

def ElkHO(N, Max_iter, lb, ub, dim, fobj):
    """
    Elk Herd Optimizer (ElkHO)

    Parameters:
        N (int): Population size.
        Max_iter (int): Maximum iterations.
        lb (float): Lower bound of variables.
        ub (float): Upper bound of variables.
        dim (int): Dimensionality of the problem.
        fobj (function): Objective function.

    Returns:
        tuple: (Best fitness, Best solution, Convergence curve)
    """
    if isinstance(ub, (int, float)):
        ub = np.full(dim, ub)
        lb = np.full(dim, lb)

    MalesRate = 0.2
    No_of_Males = int(N * MalesRate)

    Convergence_curve = np.zeros(Max_iter)

    ElkHerd = np.random.uniform(lb, ub, (N, dim))
    BestBull = np.zeros(dim)
    BestBullFitness = float('inf')

    ElkHerdFitness = np.array([fobj(ElkHerd[i, :]) for i in range(N)])

    for l in range(Max_iter):
        sorted_indexes = np.argsort(ElkHerdFitness)
        ElkHerd = ElkHerd[sorted_indexes, :]
        ElkHerdFitness = ElkHerdFitness[sorted_indexes]

        BestBull = ElkHerd[0, :]
        BestBullFitness = ElkHerdFitness[0]

        TransposeFitness = 1 / ElkHerdFitness[:No_of_Males]

        Families = np.zeros(N, dtype=int)
        for i in range(No_of_Males, N):
            randNumber = np.random.rand()
            MaleIndex = 0
            sum_fitness = 0.0

            for j in range(No_of_Males):
                sum_fitness += TransposeFitness[j] / np.sum(TransposeFitness)
                if sum_fitness > randNumber:
                    MaleIndex = j
                    break

            Families[i] = sorted_indexes[MaleIndex]

        NewElkHerd = ElkHerd.copy()
        for i in range(N):
            if Families[i] == 0:
                h = np.random.randint(0, N)
                for j in range(dim):
                    NewElkHerd[i, j] = ElkHerd[i, j] + np.random.rand() * (ElkHerd[h, j] - ElkHerd[i, j])
                    NewElkHerd[i, j] = np.clip(NewElkHerd[i, j], lb[j], ub[j])
            else:
                h = np.random.randint(0, N)
                MaleIndex = Families[i]
                hh = np.random.permutation(np.sum(Families == MaleIndex))
                h = np.random.randint(1, len(hh))
                for j in range(dim):
                    rd = -2 + 4 * np.random.rand()
                    NewElkHerd[i, j] = ElkHerd[i, j] + (ElkHerd[Families[i], j] - ElkHerd[i, j]) + rd * (ElkHerd[h, j] - ElkHerd[i, j])

        NewElkHerdFitness = np.array([fobj(NewElkHerd[i, :]) for i in range(N)])
        for i in range(N):
            if NewElkHerdFitness[i] < BestBullFitness:
                BestBull = NewElkHerd[i, :]
                BestBullFitness = NewElkHerdFitness[i]

        combined_population = np.vstack((ElkHerd, NewElkHerd))
        combined_fitness = np.array([fobj(combined_population[i, :]) for i in range(combined_population.shape[0])])

        sorted_indexes = np.argsort(combined_fitness)
        ElkHerd = combined_population[sorted_indexes[:N], :]
        ElkHerdFitness = combined_fitness[sorted_indexes[:N]]

        Convergence_curve[l] = BestBullFitness

    return BestBullFitness, BestBull, Convergence_curve
