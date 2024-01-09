#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 16:00:41 2024

Convergence plots (on Dahlquist) for the article
"""
import os
import numpy as np
from utils import getParamsRK, getParamsSDC, solutionSDC, \
    plt

PATH = '/'+os.path.join(*__file__.split('/')[:-1])
SCRIPT = __file__.split('/')[-1].split('.')[0]

# Script parameters
lam = 1j
tEnd = 2*np.pi
nStepsList = np.array([2, 5, 10, 20, 50, 100, 200, 500, 1000])
dtVals = tEnd/nStepsList

# Collocation parameters
nodeType = "LEGENDRE"

def getError(uNum, uRef):
    if uNum is None:
        return np.inf
    return np.linalg.norm(uRef - uNum[:, 0], np.inf)

# Configuration
# (nNodes, quadType, sweepType)
config = [
    (4, "RADAU-RIGHT", "MIN-SR-NS"),
    (5, "LOBATTO", "MIN-SR-NS"),
    (4, "RADAU-RIGHT", "MIN-SR-S"),
    (5, "LOBATTO", "MIN-SR-S"),
    (4, "RADAU-RIGHT", "MIN-SR-FLEX"),
    (5, "LOBATTO", "MIN-SR-FLEX"),
    ]

# -----------------------------------------------------------------------------
# Script execution
# -----------------------------------------------------------------------------
for nNodes, quadType, sweepType in config:

    # Schemes parameters
    schemes = [
        ("RK4", None), ("ESDIRK53", None),
        *[(sweepType, i) for i in range(1, nNodes+1)]
    ]

    # Plot styles
    styles = [
        dict(ls="--", c="gray"), dict(ls="-.", c="gray"),
        dict(ls="-", marker='o'),
        dict(ls="-", marker='>'),
        dict(ls="-", marker='s'),
        dict(ls="-", marker='^'),
        dict(ls="-", marker='*')
    ]

    # Figure generation
    figName = f"{sweepType}_{quadType}"
    plt.figure(f"{sweepType}_{quadType}")
    for (qDelta, nSweeps), style in zip(schemes, styles):

        if nSweeps is None:
            params = getParamsRK(qDelta)
            label = None
        else:
            params = getParamsSDC(quadType, nNodes, qDelta, nSweeps, nodeType)
            label = f"$K={nSweeps}$"
        errors = []

        for nSteps in nStepsList:

            uNum, counters, parallel = solutionSDC(
                tEnd, nSteps, params, 'DAHLQUIST', lambdas=np.array([lam]))

            tVals = np.linspace(0, tEnd, nSteps+1)
            uExact = np.exp(lam*tVals)

            err = getError(uNum, uExact)
            errors.append(err)

        plt.loglog(dtVals, errors, **style, label=label)

    plt.legend()
    plt.xlabel(r"$\Delta{t}$")
    plt.ylabel(r"$L_\infty$ error")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{PATH}/{SCRIPT}_{figName}.pdf")