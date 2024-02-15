#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 11:02:39 2023

Script to investigate diagonal SDC on the ProtheroRobinson
(linear and non-linear) problem :

- error VS time-step
- error VS computation cost

Note : implementation in progress ...
"""
import numpy as np
import matplotlib.pyplot as plt

from pySDC.projects.parallelSDC_reloaded.utils import getParamsSDC, getParamsRK, solutionSDC, solutionExact

# Problem parameters
tEnd = 1
pName = "KAPS"
pParams = {
    "epsilon": 1e-6,
}


def getError(uNum, uRef):
    if uNum is None:
        return np.inf
    return max(np.linalg.norm(uRef[:, 0] - uNum[:, 0], np.inf), np.linalg.norm(uRef[:, 1] - uNum[:, 1], np.inf))


def getCost(counters):
    nNewton, nRHS, tComp = counters
    return nNewton + nRHS


# Base variable parameters
nNodes = 4
quadType = 'RADAU-RIGHT'
nodeType = 'LEGENDRE'
parEfficiency = 0.8  # 1/nNodes

qDeltaList = [
    'RK4',
    'ESDIRK53',
    'DIRK43',
    # 'IE', 'LU', 'IEpar', 'PIC',
    'MIN-SR-NS',
    'MIN-SR-S',
    'MIN-SR-FLEX',
    # "MIN3",
]
nStepsList = np.array([2, 5, 10, 20, 50, 100])
nSweepList = [1, 2, 3, 4, 5, 6]

# qDeltaList = ['MIN-SR-S']
nSweepList = [4]


symList = ['o', '^', 's', '>', '*', '<', 'p', '>'] * 10
fig, axs = plt.subplots(1, 2)

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

            uRef = solutionExact(tEnd, nSteps, pName, **pParams)

            uSDC, counters, parallel = solutionSDC(tEnd, nSteps, params, pName, **pParams)

            err = getError(uSDC, uRef)
            errors.append(err)

            cost = getCost(counters)
            if parallel:
                cost /= nNodes * parEfficiency
            costs.append(cost)

        # error VS dt
        axs[0].loglog(dtVals, errors, sym + '-', label=name)
        # error VS cost
        axs[1].loglog(costs, errors, sym + '-', label=name)

for i in range(2):
    axs[i].set(
        xlabel=r"$\Delta{t}$" if i == 0 else "cost",
        ylabel=r"$L_\infty$ error",
        ylim=(1e-17, 1e0),
    )
    axs[i].legend(loc="lower right" if i == 0 else "lower left")
    axs[i].grid()

fig.set_size_inches(12, 5)
fig.tight_layout()
