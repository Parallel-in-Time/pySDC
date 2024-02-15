#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 13:12:40 2023

Setup script for the ProtheroRobinson (linear and non-linear) problem
"""
import numpy as np
import matplotlib.pyplot as plt

from pySDC.projects.parallelSDC_reloaded.utils import solutionExact, getParamsRK, solutionSDC

script = __file__.split('/')[-1].split('.')[0]

tEnd = 2 * np.pi
nSteps = 2
epsilon = np.inf
rkScheme = "ESDIRK43"

tVals = np.linspace(0, tEnd, nSteps + 1)
tValsFine = np.linspace(0, tEnd, nSteps * 10 + 1)

for pType in ["linear", "nonlinear"]:
    probName = "PROTHERO-ROBINSON"
    if pType == "nonlinear":
        probName += "-NL"

    print(f"Computing {pType} ODE solution")
    uExact = solutionExact(tEnd, nSteps, "PROTHERO-ROBINSON", epsilon=epsilon)

    params = getParamsRK(rkScheme)
    uNum, counters, parallel = solutionSDC(tEnd, nSteps, params, probName, epsilon=epsilon)
    uNumFine, counters, parallel = solutionSDC(tEnd, nSteps * 10, params, probName, epsilon=epsilon)

    figName = f"{script}_{pType}"
    plt.figure(figName)
    plt.plot(tVals, uExact[:, 0], '-', label="Exact")
    plt.plot(tVals, uNum[:, 0], '--', label="Numerical")
    plt.plot(tValsFine, uNumFine[:, 0], '-.', label="Numerical (fine)")


for figName in [f"{script}_linear", f"{script}_nonlinear"]:
    plt.figure(figName)
    plt.legend()
    plt.xlabel("time")
    plt.ylabel("solution")
    plt.tight_layout()
