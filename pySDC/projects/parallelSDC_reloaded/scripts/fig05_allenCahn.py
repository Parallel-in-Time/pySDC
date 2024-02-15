#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 11:14:01 2024

Figures with experiments on the Allen-Cahn problem
"""
import os
import numpy as np

from pySDC.projects.parallelSDC_reloaded.utils import solutionExact, getParamsSDC, solutionSDC, getParamsRK, plt
from pySDC.helpers.testing import DataChecker

data = DataChecker(__file__)

PATH = '/' + os.path.join(*__file__.split('/')[:-1])
SCRIPT = __file__.split('/')[-1].split('.')[0]

symList = ['o', '^', 's', '>', '*', '<', 'p', '>'] * 10

# SDC parameters
nNodes = 4
quadType = 'RADAU-RIGHT'
nodeType = 'LEGENDRE'
parEfficiency = 0.8  # 1/nNodes
nSweeps = 4

# Problem parameters
pName = "ALLEN-CAHN"
tEnd = 50
pParams = {
    "periodic": False,
    "nvars": 2**11 - 1,
    "epsilon": 0.04,
}

# -----------------------------------------------------------------------------
# Trajectories (reference solution)
# -----------------------------------------------------------------------------
uExact = solutionExact(tEnd, 1, pName, **pParams)
x = np.linspace(-0.5, 0.5, 2**11 + 1)[1:-1]

figName = f"{SCRIPT}_solution"
plt.figure(figName)
plt.plot(x, uExact[0, :], '-', label="$u(0)$")
plt.plot(x, uExact[-1, :], '--', label="$u(T)$")

plt.legend()
plt.xlabel("$x$")
plt.ylabel("Solution")
plt.gcf().set_size_inches(12, 3)
plt.tight_layout()
plt.savefig(f"{PATH}/{figName}.pdf")

# -----------------------------------------------------------------------------
# %% Convergence and error VS cost plots
# -----------------------------------------------------------------------------
nStepsList = np.array([1, 2, 5, 10, 20, 50, 100, 200, 500])
dtVals = tEnd / nStepsList


def getError(uNum, uRef):
    if uNum is None:
        return np.inf
    return np.linalg.norm(uRef[-1, :] - uNum[-1, :], ord=2)


def getCost(counters):
    nNewton, nRHS, tComp = counters
    return 2 * nNewton + nRHS


minPrec = ["MIN-SR-NS", "MIN-SR-S", "MIN-SR-FLEX"]

symList = ['^', '>', '<', 'o', 's', '*', 'p']
config = [
    (*minPrec, "VDHS", "ESDIRK43", "LU"),
]


i = 0
for qDeltaList in config:
    figNameConv = f"{SCRIPT}_conv_{i}"
    figNameCost = f"{SCRIPT}_cost_{i}"
    i += 1

    for qDelta, sym in zip(qDeltaList, symList):
        try:
            params = getParamsRK(qDelta)
        except KeyError:
            params = getParamsSDC(
                quadType=quadType, numNodes=nNodes, nodeType=nodeType, qDeltaI=qDelta, nSweeps=nSweeps
            )

        errors = []
        costs = []

        for nSteps in nStepsList:
            uRef = solutionExact(tEnd, nSteps, pName, **pParams)

            uSDC, counters, parallel = solutionSDC(tEnd, nSteps, params, pName, **pParams)

            err = getError(uSDC, uRef)
            errors.append(err)

            cost = getCost(counters)
            if parallel:
                cost /= nNodes * parEfficiency
            costs.append(cost)

        ls = '-' if qDelta.startswith("MIN-SR-") else "--"

        plt.figure(figNameConv)
        plt.loglog(dtVals, errors, sym + ls, label=qDelta)
        data.storeAndCheck(f"{figNameConv}_{qDelta}", errors, atol=1e-4, rtol=1e-4)

        plt.figure(figNameCost)
        plt.loglog(costs, errors, sym + ls, label=qDelta)

    for figName in [figNameConv, figNameCost]:
        plt.figure(figName)
        plt.gca().set(
            xlabel="Cost" if "cost" in figName else r"$\Delta {t}$",
            ylabel=r"$L_2$ error at $T$",
        )
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{PATH}/{figName}.pdf")

data.writeToJSON()
