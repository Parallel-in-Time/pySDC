#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 21:22:52 2023

Setup script for the Chemical Reaction problem
"""
import numpy as np
import matplotlib.pyplot as plt

from pySDC.projects.parallelSDC_reloaded.utils import solutionExact, getParamsRK, solutionSDC

script = __file__.split('/')[-1].split('.')[0]

tEnd = 300
nSteps = 10
rkScheme = "RK4"

tVals = np.linspace(0, tEnd, nSteps + 1)

print("Computing ODE solution")
uExact = solutionExact(tEnd, nSteps, "CHEMREC")

params = getParamsRK(rkScheme)
uNum, counters, _ = solutionSDC(tEnd, nSteps, params, 'CHEMREC')

figName = f"{script}_solution"
plt.figure(figName)
plt.plot(tVals, uExact[:, 0], '-', label="c1-exact")
plt.plot(tVals, uExact[:, 1], '-', label="c2-exact")
plt.plot(tVals, uExact[:, 2], '-', label="c3-exact")
plt.plot(tVals, uNum[:, 0], '--', label="c1-num")
plt.plot(tVals, uNum[:, 1], '--', label="c2-num")
plt.plot(tVals, uNum[:, 2], '--', label="c3-num")

plt.legend()
plt.xlabel("time")
plt.ylabel("solution")
plt.tight_layout()
