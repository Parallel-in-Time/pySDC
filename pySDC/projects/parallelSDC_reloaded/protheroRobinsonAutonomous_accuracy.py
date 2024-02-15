#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 11:02:39 2023

Script to investigate diagonal SDC on the ProtheroRobinson
(linear and non-linear) problem, using the autonomous formulation.

- error VS time-step
- error VS computation cost

Note : implementation in progress ...
"""
import numpy as np
import matplotlib.pyplot as plt

from pySDC.projects.parallelSDC_reloaded.utils import getParamsSDC, getParamsRK, solutionSDC, solutionExact

# Problem parameters
tEnd = 2 * np.pi
nonLinear = True
epsilon = 0.001
collUpdate = False
initSweep = "spread"

pName = "PROTHERO-ROBINSON-A" + (nonLinear) * "-NL"


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
parEfficiency = 0.8

qDeltaList = [
    'MIN-SR-NS',
    'MIN-SR-S',
    'MIN-SR-FLEX',
    "ESDIRK43",
    'LU',
    'VDHS',
    # 'IE', 'LU', 'IEpar', 'PIC',
    # "MIN3",
]
nStepsList = np.array([2, 5, 10, 20, 50, 100, 200])
# nSweepList = [1, 2, 3, 4]

# qDeltaList = ['ESDIRK43', 'ESDIRK53', 'VDHS']
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
            name = name.split('(')[0]
            if nSweeps != nSweepList[0]:  # pragma: no cover
                continue

        except KeyError:
            params = getParamsSDC(
                quadType=quadType,
                numNodes=nNodes,
                nodeType=nodeType,
                qDeltaI=qDelta,
                nSweeps=nSweeps,
                collUpdate=collUpdate,
                initType=initSweep,
            )
        print(f'computing for {name} ...')

        errors = []
        costs = []

        for nSteps in nStepsList:
            print(f' -- nSteps={nSteps} ...')

            uRef = solutionExact(tEnd, nSteps, pName, epsilon=epsilon)

            uSDC, counters, parallel = solutionSDC(tEnd, nSteps, params, pName, epsilon=epsilon)

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
        ylim=(1e-12, 1e3),
    )
    axs[i].legend()
    axs[i].grid()

fig.set_size_inches(12, 5)
fig.tight_layout()
