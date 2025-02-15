
import numpy as np

def ALO(N, Max_iter, lb, ub, dim, fobj):
    antlion_position = np.random.rand(N, dim) * (ub - lb) + lb
    ant_position = np.random.rand(N, dim) * (ub - lb) + lb

    antlions_fitness = np.array([fobj(antlion_position[i, :]) for i in range(N)])
    sorted_indexes = np.argsort(antlions_fitness)
    sorted_antlions = antlion_position[sorted_indexes, :]

    Elite_antlion_position = sorted_antlions[0, :]
    Elite_antlion_fitness = antlions_fitness[sorted_indexes[0]]

    Convergence_curve = np.zeros(Max_iter)

    for It in range(1, Max_iter):
        for i in range(N):
            Rolette_index = np.random.randint(0, N)
            RA = np.random.rand(dim) * (ub - lb) + lb
            RE = np.random.rand(dim) * (ub - lb) + lb
            ant_position[i, :] = (RA + RE) / 2

        ants_fitness = np.array([fobj(ant_position[i, :]) for i in range(N)])
        double_population = np.vstack((sorted_antlions, ant_position))
        double_fitness = np.hstack((antlions_fitness, ants_fitness))

        sorted_indexes = np.argsort(double_fitness)
        sorted_antlions = double_population[sorted_indexes[:N], :]
        antlions_fitness = double_fitness[sorted_indexes[:N]]

        if antlions_fitness[0] < Elite_antlion_fitness:
            Elite_antlion_position = sorted_antlions[0, :]
            Elite_antlion_fitness = antlions_fitness[0]

        Convergence_curve[It] = Elite_antlion_fitness

    return Elite_antlion_fitness, Elite_antlion_position, Convergence_curve
