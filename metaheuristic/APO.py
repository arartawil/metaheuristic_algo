
import numpy as np

def APO(pop_size, iter_max, Xmin, Xmax, dim, fhd):
    protozoa = Xmin + np.random.rand(pop_size, dim) * (Xmax - Xmin)
    protozoa_Fit = np.array([fhd(protozoa[i, :]) for i in range(pop_size)])

    bestFit = np.min(protozoa_Fit)
    bestProtozoa = protozoa[np.argmin(protozoa_Fit), :]

    f_out_convergence = np.zeros(iter_max)

    for iter in range(1, iter_max):
        index = np.argsort(protozoa_Fit)
        protozoa = protozoa[index, :]
        protozoa_Fit = protozoa_Fit[index]

        pf = 0.1 * np.random.rand()
        ri = np.random.choice(pop_size, size=int(pop_size * pf), replace=False)

        for i in range(pop_size):
            if i in ri:
                newprotozoa = Xmin + np.random.rand(dim) * (Xmax - Xmin)
            else:
                f = np.random.rand() * (1 + np.cos(iter / iter_max * np.pi))
                newprotozoa = protozoa[i, :] + f * (protozoa[np.random.randint(pop_size), :] - protozoa[i, :])

            newprotozoa = np.clip(newprotozoa, Xmin, Xmax)
            newprotozoa_Fit = fhd(newprotozoa)

            if newprotozoa_Fit < protozoa_Fit[i]:
                protozoa[i, :] = newprotozoa
                protozoa_Fit[i] = newprotozoa_Fit

        bestFit = np.min(protozoa_Fit)
        bestProtozoa = protozoa[np.argmin(protozoa_Fit), :]
        f_out_convergence[iter] = bestFit

    return bestFit, bestProtozoa, f_out_convergence
