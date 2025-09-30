#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 30 13:02:50 2025

@author: cpf5546
"""
import os
import matplotlib.pyplot as plt

from pySDC.playgrounds.dedalus.timestepper import SDCIMEX
from pySDC.playgrounds.dedalus.problems.rbc import RBCProblem3D, OutputFiles


dtBase = 2.50e-03
nSteps = 10
nCycles = 2

Rayleigh = 1.5e5
timeScheme = "SDC"
stabDir = f"stab_A4_M1_R1_{timeScheme}"
initSol = "init_3D_A4_M1_R1.pySDC"

SDCIMEX.setParameters(
    nNodes=4, nodeType="LEGENDRE", quadType="RADAU-RIGHT",
    nSweeps=4, initSweep="COPY", explSweep="PIC", implSweep="MIN-SR-FLEX",
    )

os.makedirs(stabDir, exist_ok=True)

dtFactors = [f*10**e for e in range(nCycles) for f in [1, 2, 5]]
fmtSuffix = f":0{nCycles}d"

plt.figure("spectrum")
for fac in dtFactors:
    dtRun = dtBase*fac
    runDir = f"{stabDir}/dt_f" + ("{"+fmtSuffix+"}").format(fac)

    prob = RBCProblem3D.runSimulation(
        runDir, dtRun*nSteps, dtRun, timeScheme=timeScheme,
        dtWrite=dtRun*nSteps, initField=initSol,
        aspectRatio=4, meshRatio=1, resFactor=1,
        Rayleigh=Rayleigh, Prandtl=0.7,
        )

    if "tComp" in prob.infos:
        break

    output = OutputFiles(runDir)
    spectrum = output.getSpectrum(which="all", start=1)

#     plt.loglog(spectrum["kappa"], spectrum["p"], label="f"+("{"+fmtSuffix+"}").format(fac))

# plt.legend()
