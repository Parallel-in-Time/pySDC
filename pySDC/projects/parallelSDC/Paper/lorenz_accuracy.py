#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 11:02:39 2023

Script to investigate diagonal SDC on Lorenz system with a given number of
revolution periods :

- error VS time-step
- error VS computation cost

Note : implementation in progress ...
"""
import numpy as np
import matplotlib.pyplot as plt

from utils import getParamsSDC, solutionSDC, solutionExact

tEnd = 2.82

def getError(uNum, uRef):
    if uNum is None:
        return np.inf
    return np.linalg.norm(uRef[:, 0] - uNum[:, 0], np.inf)

def getCost(counters):
    nNewton, nRHS = counters
    return nNewton + nRHS


# Base variable parameters
qDeltaList = ['LU', 'IEpar', 'MIN-SR-NS', 'MIN-SR-S', 'FLEX-MIN-1', 'MIN3']
nStepsList = np.array([2, 5, 10, 100, 200, 500, 1000])
nSweepList = [1, 2, 3, 4, 5, 6]


symList = ['o', '^', 's', '>', '*', '<', 'p', '>']*10

# qDeltaList = ['MIN-SR-NS']
nSweepList = [4]

fig, axs = plt.subplots(1, 2)


dtVals = tEnd/nStepsList

i = 0
for qDelta in qDeltaList:
    for nSweeps in nSweepList:

        sym = symList[i]
        i += 1

        name = f"{qDelta}({nSweeps})"
        print(f'computing for {name} ...')

        errors = []
        costs = []

        for nSteps in nStepsList:
            print(f' -- nSteps={nSteps} ...')

            uRef = solutionExact(tEnd, nSteps, "LORENZ", u0=(5, -5, 20))

            paramsSDC = getParamsSDC(qDeltaI=qDelta, nSweeps=nSweeps)
            uSDC, counters = solutionSDC(
                tEnd, nSteps, paramsSDC, "LORENZ", u0=(5, -5, 20))

            err = getError(uSDC, uRef)
            errors.append(err)

            cost = getCost(counters)
            costs.append(cost)

        # error VS dt
        axs[0].loglog(dtVals, errors, sym+'-', label=name)
        # error VS cost
        axs[1].loglog(costs, errors, sym+'-', label=name)

for i in range(2):
    axs[i].set(
        xlabel=r"$\Delta{t}$" if i == 0 else "cost",
        ylabel=r"$L_\infty$ error",
    )
    axs[i].legend(loc="lower right" if i == 0 else "lower left")
    axs[i].grid()

fig.set_size_inches(12, 5)
fig.tight_layout()
plt.show()
