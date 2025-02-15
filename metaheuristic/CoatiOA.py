
import numpy as np

def CoatiOA(SearchAgents, Max_iterations, lowerbound, upperbound, dimension, fitness):
    X = np.random.rand(SearchAgents, dimension) * (upperbound - lowerbound) + lowerbound
    fit = np.array([fitness(X[i, :]) for i in range(SearchAgents)])

    fbest = np.min(fit)
    Xbest = X[np.argmin(fit), :].copy()

    Convergence_curve = np.zeros(Max_iterations)

    for t in range(Max_iterations):
        for i in range(SearchAgents):
            newPos = X[i, :] + 0.1 * np.random.randn(dimension) * (Xbest - X[i, :])
            newPos = np.clip(newPos, lowerbound, upperbound)
            newFit = fitness(newPos)

            if newFit < fit[i]:
                X[i, :] = newPos
                fit[i] = newFit

            if newFit < fbest:
                Xbest = newPos.copy()
                fbest = newFit

        Convergence_curve[t] = fbest

    return fbest, Xbest, Convergence_curve
