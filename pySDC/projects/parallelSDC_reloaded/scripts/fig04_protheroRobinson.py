#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 10:21:47 2024

Figures with experiment on the Prothero-Robinson problem
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

epsilon = 1e-3

# -----------------------------------------------------------------------------
# %% Convergence and error VS cost plots
# -----------------------------------------------------------------------------
tEnd = 2 * np.pi
nStepsList = np.array([2, 5, 10, 20, 50, 100, 200, 500, 1000])
dtVals = tEnd / nStepsList


def getError(uNum, uRef):
    if uNum is None:
        return np.inf
    return np.linalg.norm(np.linalg.norm(uRef - uNum, np.inf, axis=-1), np.inf)


def getCost(counters):
    nNewton, nRHS, tComp = counters
    return nNewton + nRHS


minPrec = ["MIN-SR-NS", "MIN-SR-S", "MIN-SR-FLEX"]

symList = ['^', '>', '<', 'o', 's', '*', 'p']
config = [
    [(*minPrec, "VDHS", "ESDIRK43", "LU"), 4],
    [(*minPrec, "VDHS", "ESDIRK43", "LU"), 6],
]


i = 0
for qDeltaList, nSweeps in config:
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
            uRef = solutionExact(tEnd, nSteps, "PROTHERO-ROBINSON", epsilon=epsilon)

            uSDC, counters, parallel = solutionSDC(tEnd, nSteps, params, "PROTHERO-ROBINSON", epsilon=epsilon)

            err = getError(uSDC, uRef)
            errors.append(err)

            cost = getCost(counters)
            if parallel:
                cost /= nNodes * parEfficiency
            costs.append(cost)

        ls = '-' if qDelta.startswith("MIN-SR-") else "--"

        plt.figure(figNameConv)
        plt.loglog(dtVals, errors, sym + ls, label=qDelta)
        data.storeAndCheck(f"{figNameConv}_{qDelta}", errors)

        plt.figure(figNameCost)
        plt.loglog(costs, errors, sym + ls, label=qDelta)

    for figName in [figNameConv, figNameCost]:
        plt.figure(figName)
        plt.gca().set(
            xlabel="Cost" if "cost" in figName else r"$\Delta {t}$",
            ylabel=r"$L_\infty$ error",
        )
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{PATH}/{figName}.pdf")

data.writeToJSON()
