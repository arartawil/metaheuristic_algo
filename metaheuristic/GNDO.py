import numpy as np

def GNDO(n, t, lb, ub, d, obj):
    x = lb + (ub - lb) * np.random.rand(n, d)
    
    bestFitness = float('inf')
    bestSol = np.zeros(d)
    cgcurve = np.zeros(t)

    fitness = np.zeros(n)  

    for it in range(t):
        for i in range(n):
            fitness[i] = obj(x[i, :])  

        for i in range(n):
            if fitness[i] < bestFitness:
                bestSol = x[i,:]
                bestFitness = fitness[i]

        cgcurve[it] = bestFitness
        mo = np.mean(x, axis=0)

        for i in range(n):
            a, b, c = np.random.choice(n, 3, replace=False)

            v1 = (fitness[a] < fitness[i]) * (x[a] - x[i]) + (fitness[a] >= fitness[i]) * (x[i] - x[a])
            v2 = (fitness[b] < fitness[c]) * (x[b] - x[c]) + (fitness[b] >= fitness[c]) * (x[c] - x[b])

            if np.random.rand() <= np.random.rand():
                u = (1/3) * (x[i,:] + bestSol + mo)
                deta = np.sqrt((1/3) * ((x[i,:] - u)**2 + (bestSol - u)**2 + (mo - u)**2))

                vc1 = np.random.rand(d)
                vc2 = np.random.rand(d)
                
                Z1 = np.sqrt(-np.log(vc2)) * np.cos(2 * np.pi * vc1)
                Z2 = np.sqrt(-np.log(vc2)) * np.cos(2 * np.pi * vc1 + np.pi)
                
                a_rand = np.random.rand()
                b_rand = np.random.rand()
                
                eta = u + deta * Z1 if a_rand <= b_rand else u + deta * Z2
                newsol = eta
            else:
                beta = np.random.rand()
                v = x[i] + beta * np.abs(np.random.randn(d)) * v1 + (1 - beta) * np.abs(np.random.randn(d)) * v2
                newsol = v

            newsol = np.clip(newsol, lb, ub)
            newfitness = obj(newsol)

            if newfitness < fitness[i]:
                x[i,:] = newsol
                fitness[i] = newfitness

                if fitness[i] < bestFitness:
                    bestSol = x[i,:]
                    bestFitness = fitness[i]

        cgcurve[it] = bestFitness

    return bestFitness, bestSol, cgcurve
