#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 21:22:52 2023

Setup script for the Allen-Cahn problem
"""
import numpy as np
import matplotlib.pyplot as plt

from utils import solutionExact, getParamsRK, solutionSDC

script = __file__.split('/')[-1].split('.')[0]

tEnd = 30
nSteps = 10
rkScheme = "RK4"
pName = "ALLEN-CAHN"
pParams  = {"periodic": True}

tVals = np.linspace(0, tEnd, nSteps+1)

print("Computing ODE solution")
uExact = solutionExact(tEnd, nSteps, pName, **pParams)

# params = getParamsRK(rkScheme)
# uNum, counters = solutionSDC(tEnd, nSteps, params, probName)

figName = f"{script}_solution"
plt.figure(figName)
plt.plot(uExact[0, :], '-', label="u(0)")
plt.plot(uExact[-1, :], '-', label="u(T)")

plt.legend()
plt.xlabel("X")
plt.ylabel("solution")
plt.tight_layout()
plt.savefig(f'fig/{figName}.pdf')

# plt.show()
