#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 21:22:52 2023

Setup script for the Allen-Cahn problem
"""
import numpy as np
import matplotlib.pyplot as plt

from pySDC.projects.parallelSDC_reloaded.utils import solutionExact, getParamsRK, solutionSDC, getParamsSDC

script = __file__.split('/')[-1].split('.')[0]

tEnd = 50
nSteps = 50

useRK = True
if useRK:
    rkScheme = "ESDIRK53"
    params = getParamsRK(rkScheme)
else:  # pragma: no cover
    nNodes = 4
    nSweeps = 5
    quadType = 'RADAU-RIGHT'
    nodeType = 'LEGENDRE'
    qDelta = "MIN-SR-S"
    params = getParamsSDC(quadType, nNodes, qDelta, nSweeps, nodeType)

pName = "ALLEN-CAHN"
periodic = False
pParams = {
    "periodic": periodic,
    "nvars": 2**11 - (not periodic),
    "epsilon": 0.04,
}

tVals = np.linspace(0, tEnd, nSteps + 1)

print("Computing ODE solution")
uExact = solutionExact(tEnd, nSteps, pName, **pParams)


uNum, counters, _ = solutionSDC(tEnd, nSteps, params, pName, **pParams)

figName = f"{script}_solution"
plt.figure(figName)
plt.plot(uExact[0, :], '-', label="$u(0)$")
plt.plot(uExact[-1, :], '-', label="$u_{exact}(T)$")
plt.plot(uNum[-1, :], '--', label="$u_{num}(T)$")


plt.legend()
plt.xlabel("X")
plt.ylabel("solution")
plt.tight_layout()
