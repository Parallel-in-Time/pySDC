#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 22:14:03 2023

Script to investigate diagonal SDC on Van der Pol with different mu parameters,
in particular with graphs such as :

- accuracy VS time-step
- accuracy VS rhs evaluation

Note : implementation in progress ...
"""
import numpy as np

from utils import getParamsSDC, solVanderpolSDC, solVanderpolExact

muVals = [0.1, 1, 10]
tEndVals = [6.3, 6.7, 18.9] # 1 periods for each mu

def getError(uNum, uRef):
    if uNum is None:
        return np.inf
    return np.linalg.norm(uRef[:, 0] - uNum[:, 0], np.inf)


qDeltaList = ['LU', 'IEpar', 'MIN-SR-NS', 'MIN-SR-S', 'FLEX-MIN-1', 'MIN3']


for mu, tEnd in zip(muVals, tEndVals):
    print("-"*80)
    print(f"mu={mu}")
    print("-"*80)

    for nSteps in [1, 10, 100, 200, 500]:
        print(f"nSteps={nSteps}")

        uRef = solVanderpolExact(tEnd, nSteps, mu=mu)

        for qDeltaI in qDeltaList:
            paramsSDC = getParamsSDC(qDeltaI=qDeltaI)
            uSDC_LU = solVanderpolSDC(tEnd, nSteps, paramsSDC, mu=mu)

            err = getError(uSDC_LU, uRef)
            print(f"  {qDeltaI} -- err={err:1.2g}")
