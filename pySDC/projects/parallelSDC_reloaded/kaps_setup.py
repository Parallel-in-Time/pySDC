#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 19:29:46 2023

Setup script for the Kaps problem
"""
import numpy as np
import matplotlib.pyplot as plt

from pySDC.projects.parallelSDC_reloaded.utils import solutionExact, getParamsRK, solutionSDC

script = __file__.split('/')[-1].split('.')[0]

tEnd = 1
nSteps = 100
epsilon = 1e-3
rkScheme = "DIRK43"

tVals = np.linspace(0, tEnd, nSteps + 1)

print("Computing ODE solution")
uExact = solutionExact(tEnd, nSteps, "KAPS", epsilon=epsilon)

params = getParamsRK(rkScheme)
uNum, counters, parallel = solutionSDC(tEnd, nSteps, params, 'KAPS', epsilon=epsilon)

figName = f"{script}_solution"
plt.figure(figName)
plt.plot(tVals, uExact[:, 0], '-', label="x-exact")
plt.plot(tVals, uExact[:, 1], '-', label="y-exact")
plt.plot(tVals, uNum[:, 0], '--', label="x-num")
plt.plot(tVals, uNum[:, 1], '--', label="y-num")

plt.legend()
plt.xlabel("time")
plt.ylabel("solution")
plt.tight_layout()
