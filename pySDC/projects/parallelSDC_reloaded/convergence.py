#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 14:44:41 2024

Generate convergence plots on Dahlquist for SDC with given parameters
"""
import numpy as np
from pySDC.projects.parallelSDC_reloaded.utils import getParamsRK, getParamsSDC, solutionSDC, plt

SCRIPT = __file__.split('/')[-1].split('.')[0]

# Script parameters
lam = 1j
tEnd = 2 * np.pi
nStepsList = np.array([2, 5, 10, 20, 50, 100, 200, 500, 1000])
dtVals = tEnd / nStepsList


def getError(uNum, uRef):
    if uNum is None:  # pragma: no cover
        return np.inf
    return np.linalg.norm(uRef - uNum[:, 0], np.inf)


# Collocation parameters
nNodes = 4
nodeType = "LEGENDRE"
quadType = "RADAU-RIGHT"
sweepType = "MIN-SR-NS"

# Schemes parameters
schemes = [("RK4", None), ("ESDIRK43", None), *[(sweepType, i) for i in [1, 2, 3, 4]][:1]]

styles = [
    dict(ls=":", c="gray"),
    dict(ls="-.", c="gray"),
    dict(ls="-", marker='o'),
    dict(ls="-", marker='>'),
    dict(ls="-", marker='s'),
    dict(ls="-", marker='^'),
    dict(ls="-", marker='*'),
]

# -----------------------------------------------------------------------------
# Script execution
# -----------------------------------------------------------------------------
plt.figure()
for (qDelta, nSweeps), style in zip(schemes, styles):
    if nSweeps is None:
        params = getParamsRK(qDelta)
        label = None
    else:
        params = getParamsSDC(quadType, nNodes, qDelta, nSweeps, nodeType)
        label = f"$K={nSweeps}$"
    errors = []

    for nSteps in nStepsList:
        uNum, counters, parallel = solutionSDC(tEnd, nSteps, params, 'DAHLQUIST', lambdas=np.array([lam]))

        tVals = np.linspace(0, tEnd, nSteps + 1)
        uExact = np.exp(lam * tVals)

        err = getError(uNum, uExact)
        errors.append(err)

    plt.loglog(dtVals, errors, **style, label=label)
    if nSweeps is not None:
        plt.loglog(dtVals, (0.1 * dtVals) ** nSweeps, '--', c='gray', lw=1.5)

plt.title(sweepType)
plt.legend()
plt.xlabel(r"$\Delta{t}$")
plt.ylabel(r"$L_\infty$ error")
plt.grid(True)
plt.tight_layout()
