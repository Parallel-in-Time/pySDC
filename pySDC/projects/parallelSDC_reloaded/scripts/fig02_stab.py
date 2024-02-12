#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 12:02:55 2024

Stability plots (on Dahlquist) for the article
"""
import os
import numpy as np

from pySDC.projects.parallelSDC_reloaded.utils import getParamsSDC, solutionSDC, plotStabContour, plt
from pySDC.helpers.testing import DataChecker

data = DataChecker(__file__)

PATH = '/' + os.path.join(*__file__.split('/')[:-1])
SCRIPT = __file__.split('/')[-1].split('.')[0]

# Script parameters
zoom = 2
reLims = -4.5 * zoom, 0.5 * zoom
imLims = -3.5 * zoom, 3.5 * zoom
nVals = 251

# Collocation parameters
nNodes = 4
nodeType = "LEGENDRE"
quadType = "RADAU-RIGHT"

# Configuration
# (qDeltaType)
config = [
    "PIC",
    "MIN-SR-NS",
    "MIN-SR-S",
    "MIN-SR-FLEX",
    "LU",
    "VDHS",
]


# -----------------------------------------------------------------------------
# Script execution
# -----------------------------------------------------------------------------

# Problem instanciation
reVals = np.linspace(*reLims, num=nVals)
imVals = np.linspace(*imLims, num=nVals)
lambdas = reVals[None, :] + 1j * imVals[:, None]

# Scheme instanciation
for qDeltaType in config:
    if qDeltaType == "MIN-SR-S":
        fac = 5
        reVals *= fac
        imVals *= fac
        lambdas *= fac

    for nSweeps in [1, 2, 3, 4]:
        params = getParamsSDC(quadType, nNodes, qDeltaType, nSweeps, nodeType)

        uNum, counters, parallel = solutionSDC(1, 1, params, 'DAHLQUIST', lambdas=lambdas.ravel())

        uEnd = uNum[-1, :].reshape(lambdas.shape)
        stab = np.abs(uEnd)

        figName = f"{qDeltaType}_K{nSweeps}"
        plt.figure(figName)

        plotStabContour(reVals, imVals, stab)
        data.storeAndCheck(f"{SCRIPT}_{figName}", stab[::5, -50])

        plt.xticks(fontsize=8)
        plt.yticks(fontsize=8)
        plt.title(f"$K={nSweeps}$", fontsize=10)
        plt.gcf().set_size_inches(2.5, 2.5)
        plt.tight_layout()
        plt.savefig(f"{PATH}/{SCRIPT}_{figName}.pdf")

data.writeToJSON()
