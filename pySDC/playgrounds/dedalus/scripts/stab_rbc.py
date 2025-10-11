#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 30 13:02:50 2025

@author: cpf5546
"""
import os
import matplotlib.pyplot as plt
import numpy as np

from pySDC.playgrounds.dedalus.timestepper import SDCIMEX
from pySDC.playgrounds.dedalus.problems.rbc import RBCProblem3D, OutputFiles

tEnd = 10

nStepsMax = 30
nStepsMin = 12
decrement = 1

nStepsVals = np.arange(nStepsMax, nStepsMin-1, -abs(decrement))
dtVals = 1/nStepsVals


Rayleigh = 1.5e5
timeScheme = "RK443"
stabDir = f"stab_A4_M1_R1_{timeScheme}"
initSol = "init_3D_A4_M1_R1.pySDC"

SDCIMEX.setParameters(
    nNodes=4, nodeType="LEGENDRE", quadType="RADAU-RIGHT",
    nSweeps=4, initSweep="COPY", explSweep="PIC", implSweep="MIN-SR-FLEX",
    )

os.makedirs(stabDir, exist_ok=True)

fmtSuffix = f":0{len(str(nStepsMax))}d"

plt.figure("spectrum")
for nSteps in nStepsVals:
    dtRun = 1/nSteps
    runDir = f"{stabDir}/dt_N" + ("{"+fmtSuffix+"}").format(nSteps)

    prob = RBCProblem3D.runSimulation(
        runDir, tEnd, dtRun, timeScheme=timeScheme,
        dtWrite=tEnd, initField=initSol,
        aspectRatio=4, meshRatio=1, resFactor=1,
        Rayleigh=Rayleigh, Prandtl=0.7,
        )

    if "tComp" in prob.infos:
        break

    output = OutputFiles(runDir)
    spectrum = output.getSpectrum(which="all", start=1)

    if np.any(np.isnan(spectrum["u"])):
        break
    plt.loglog(spectrum["kappa"], spectrum["u"], ':', label="N"+("{"+fmtSuffix+"}").format(nSteps))

plt.legend()
