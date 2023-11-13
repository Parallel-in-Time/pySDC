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
import matplotlib.pyplot as plt

from utils import getParamsSDC, solVanderpolSDC, solVanderpolExact

muVals = [0.1, 2, 10]
tEndVals = [6.3, 7.6, 18.9] # 1 periods for each mu

def getError(uNum, uRef):
    if uNum is None:
        return np.inf
    return np.linalg.norm(uRef[:, 0] - uNum[:, 0], np.inf)

# Base variable parameters
qDeltaList = ['LU', 'IEpar', 'MIN-SR-NS', 'MIN-SR-S', 'FLEX-MIN-1', 'MIN3']
nStepsList = np.array([2, 5, 10, 100, 200, 500, 1000])
nSweepList = [1, 2, 3, 4]


nSweepList = [3]

for mu, tEnd in zip(muVals, tEndVals):
    print("-"*80)
    print(f"mu={mu}")
    print("-"*80)

    dtVals = tEnd/nStepsList

    plt.figure(f'mu={mu}')
    for qDelta in qDeltaList:
        for nSweeps in nSweepList:
            name = f"{qDelta}({nSweeps})"
            print(f'computing for {name} ...')

            errors = []

            for nSteps in nStepsList:
                print(f' -- nSteps={nSteps} ...')

                uRef = solVanderpolExact(tEnd, nSteps, mu=mu)

                paramsSDC = getParamsSDC(qDeltaI=qDelta, nSweeps=nSweeps)
                uSDC = solVanderpolSDC(tEnd, nSteps, paramsSDC, mu=mu)

                err = getError(uSDC, uRef)
                # some hack for very stiff configurations
                if mu == 10 and nSteps == 2:
                    err = np.inf
                errors.append(err)

            plt.loglog(dtVals, errors, 'o-', label=name)
    plt.legend()
    plt.xlabel(r"$\Delta{t}$")
    plt.ylabel(r"$L_\infty$ error")
    plt.grid()
