#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 21:22:52 2023

Setup script for the JacobianElliptic problem
"""
import numpy as np
import matplotlib.pyplot as plt

from pySDC.projects.parallelSDC_reloaded.utils import solutionExact, getParamsRK, getParamsSDC, solutionSDC

script = __file__.split('/')[-1].split('.')[0]

tEnd = 10
nSteps = 100

useRK = True
if useRK:
    rkScheme = "RK4"
    params = getParamsRK(rkScheme)
else:  # pragma: no cover
    nNodes = 4
    nSweeps = 5
    quadType = 'RADAU-RIGHT'
    nodeType = 'LEGENDRE'
    qDelta = "MIN-SR-S"
    params = getParamsSDC(quadType, nNodes, qDelta, nSweeps, nodeType)

pName = "JACELL"
periodic = False
pParams = {}

tVals = np.linspace(0, tEnd, nSteps + 1)

print("Computing ODE solution")
uExact = solutionExact(tEnd, nSteps, pName, **pParams)

uNum, counters, _ = solutionSDC(tEnd, nSteps, params, pName, **pParams)

figName = f"{script}_solution"
plt.figure(figName)
plt.plot(tVals, uExact[:, 0], '-', label="u1-exact")
plt.plot(tVals, uExact[:, 1], '-', label="u2-exact")
plt.plot(tVals, uExact[:, 2], '-', label="u3-exact")
plt.plot(tVals, uNum[:, 0], '--', label="u1-num")
plt.plot(tVals, uNum[:, 1], '--', label="u2-num")
plt.plot(tVals, uNum[:, 2], '--', label="u3-num")

plt.legend()
plt.xlabel("time")
plt.ylabel("solution")
plt.tight_layout()
