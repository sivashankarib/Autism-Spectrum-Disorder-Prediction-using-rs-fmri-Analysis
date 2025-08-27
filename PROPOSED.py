import random as rn
from math import ceil, sin, exp, pi
import numpy as np
import time


# Position update is done at line 67
# Modified Cheetah Optimizer (MCO)
def PROPOSED(Positions, fobj, VRmin, VRmax, Max_iter):
    N, dim = Positions.shape[0], Positions.shape[1]
    lb = VRmin
    ub = VRmax
    BestSol = np.zeros((dim))
    BestCost = float('inf')
    Globest = BestCost
    Position = lb + np.random.rand(dim) * (ub - lb)
    Cost = np.zeros(N)
    for i in range(N):
        # for j in range(dim):
        Cost[i] = fobj(Position[i])
        if Cost[i] < BestCost:
            BestSol = Position.copy()
            BestCost = Cost[i]

    Convergence_curve = np.zeros((Max_iter, 1))
    pop1 = BestCost
    X_best = BestSol
    Globest = BestCost

    it = 1
    T = ceil(Positions[0, 0] / 10) * 60
    FEs = 0
    t = 0
    ct = time.time()

    while FEs <= Max_iter:
        dl = np.random.rand(dim, N)
        for k in range(dim):
            i = dl[k]

            if k == len(dl) - 1:
                a = dl[k - 1]
            else:
                a = dl[k + 1]

            X = Positions
            X1 = Positions[a.astype('int'), k]
            Xb = BestSol
            Xbest = BestSol

            xd = np.zeros((X.shape[0], X.shape[1]))
            if (it % 100) == 0 or it == 1:
                xd = np.random.permutation((X.shape[0]))

            Z = X.copy()
            fitness = np.zeros(N)
            for j in range(len(xd)):
                r_Hat = rn.random()
                r1 = rn.random()
                if k == 1:
                    alpha = 0.0001 * t / T * (ub[j] - lb[j])
                else:
                    alpha = 0.0001 * t / T * abs(Xb[k] - X[j, k]) + 0.001 * round(rn.random() > 0.9)
                # Update the position using this variable
                # r = rn.random()
                r = Cost[j] / ((np.min(Cost) / np.max(Cost)) / (np.max(Cost) / np.min(Cost)))

                r_Check = abs(r) ** exp(r / 2) * sin(2 * pi * r)
                beta = X1[j] - X[j, k]

                h0 = exp(2 - 2 * t / T)
                H = abs(2 * r1 * h0 - h0)

                r2 = rn.random()
                r3 = k + rn.random()

                if r2 <= r3:
                    r4 = 3 * rn.random()
                    if H > r4:
                        Z[j] = X[j] + r_Hat ** -1 * alpha
                    else:
                        Z[j] = Xbest[k] + r_Check * beta
                else:
                    Z[j] = X[j]

            xx1 = int(np.var(Z[(i.astype(int))] < lb))
            # Z[xx1] = lb[xx1] + np.random.rand((xx1)) * (ub[xx1] - lb[xx1])
            Z[i.astype(int)] = VRmin[i.astype(int)] + np.random.rand(dim) * (
                        VRmax[i.astype(int)] - VRmin[i.astype(int)])
            xx1 = int(np.var(Z[(i.astype(int))] < ub))
            # Z[xx1] = lb[xx1] + np.random.rand((xx1)) * (ub[xx1] - lb[xx1])
            Z[i.astype(int)] = VRmax[i.astype(int)] + np.random.rand(dim) * (
                        VRmin[i.astype(int)] - VRmax[i.astype(int)])

            Positions = Z.copy()
            fitness[j] = fobj(Positions[j, :])
            if fitness[j] < BestCost:
                BestCost = fitness[j]
                BestSol = Positions[j, :]

            FEs = FEs + 1

        t = t + 1

        if t > T and t - round(T) - 1 >= 1 and t > 2:
            if abs(BestCost[t - 1] - BestCost[(t - round(T) - 1)]) <= abs(0.01 * BestCost[t - 1]):
                best = BestSol.Position
                jn = np.random.rand(Positions, 1, ceil(Positions / 10 * rn.random()))
                best[jn] = lb(jn) + np.random.rand(1, len(jn)) * (ub(jn) - lb(jn))
                BestSol.Cost = fobj(best)
                BestSol.Position = best
                FEs = FEs + 1

                i0 = np.random.rand(N, round(1 * N))

                BestCost[i0[N - dim + 1: N]] = pop1[dl[1: dim]]

                BestCost = X_best

                t = 1

        it = it + 1
        Convergence_curve[t] = np.min(X_best)
        t = t + 1
    X_best = Convergence_curve[Max_iter - 1][0]
    ct = time.time() - ct
    X_best = BestSol
    return Globest, Convergence_curve, X_best, ct
