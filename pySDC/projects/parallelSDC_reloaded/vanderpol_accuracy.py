#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 22:14:03 2023

Script to investigate diagonal SDC on Van der Pol with different mu parameters,
in particular with graphs such as :

- error VS time-step
- error VS computation cost

Note : implementation in progress ...
"""
import numpy as np
import matplotlib.pyplot as plt

from pySDC.projects.parallelSDC_reloaded.utils import getParamsSDC, getParamsRK, solutionSDC, solutionExact

muVals = [0.1, 2, 10]
tEndVals = [6.3, 7.6, 18.9]  # tEnd = 1 period for each mu


def getError(uNum, uRef):
    if uNum is None:
        return np.inf
    return np.linalg.norm(uRef[:, 0] - uNum[:, 0], np.inf)


def getCost(counters):
    nNewton, nRHS, tComp = counters
    return nNewton + nRHS


# Base variable parameters
nNodes = 4
quadType = 'RADAU-RIGHT'
nodeType = 'LEGENDRE'
parEfficiency = 1 / nNodes

qDeltaList = [
    'RK4',
    'ESDIRK43',
    'LU',
    # 'IE', 'LU', 'IEpar', 'PIC',
    'MIN-SR-NS',
    'MIN-SR-S',
    'MIN-SR-FLEX',
]
nStepsList = np.array([2, 5, 10, 20, 50, 100, 200])
nSweepList = [1, 2, 3, 4, 5, 6]


symList = ['o', '^', 's', '>', '*', '<', 'p', '>'] * 10

# qDeltaList = ['LU']
nSweepList = [4]

fig, axs = plt.subplots(2, len(muVals))

for j, (mu, tEnd) in enumerate(zip(muVals, tEndVals)):
    print("-" * 80)
    print(f"mu={mu}")
    print("-" * 80)

    dtVals = tEnd / nStepsList

    i = 0
    for qDelta in qDeltaList:
        for nSweeps in nSweepList:
            sym = symList[i]
            i += 1

            name = f"{qDelta}({nSweeps})"
            try:
                params = getParamsRK(qDelta)
                name = name[:-3]
            except KeyError:
                params = getParamsSDC(
                    quadType=quadType, numNodes=nNodes, nodeType=nodeType, qDeltaI=qDelta, nSweeps=nSweeps
                )
            print(f'computing for {name} ...')

            errors = []
            costs = []

            for nSteps in nStepsList:
                print(f' -- nSteps={nSteps} ...')

                uRef = solutionExact(tEnd, nSteps, "VANDERPOL", mu=mu)

                uSDC, counters, parallel = solutionSDC(tEnd, nSteps, params, "VANDERPOL", mu=mu)

                err = getError(uSDC, uRef)
                errors.append(err)

                cost = getCost(counters)
                if parallel:
                    cost /= nNodes * parEfficiency
                costs.append(cost)

            # error VS dt
            axs[0, j].loglog(dtVals, errors, sym + '-', label=name)
            # error VS cost
            axs[1, j].loglog(costs, errors, sym + '-', label=name)

    for i in range(2):
        if i == 0:
            axs[i, j].set_title(f"mu={mu}")
        axs[i, j].set(
            xlabel=r"$\Delta{t}$" if i == 0 else "cost",
            ylabel=r"$L_\infty$ error",
            ylim=(1e-11, 10),
        )
        axs[i, j].legend(loc="lower right" if i == 0 else "lower left")
        axs[i, j].grid()

fig.set_size_inches(18.2, 10.4)
fig.tight_layout()
