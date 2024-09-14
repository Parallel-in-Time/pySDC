#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 11:14:01 2024

Figures with experiments on the Allen-Cahn problem (MPI runs)
"""
import os
import sys
import json
import numpy as np
from mpi4py import MPI

from pySDC.projects.parallelSDC_reloaded.utils import solutionExact, getParamsSDC, solutionSDC, getParamsRK
from pySDC.implementations.sweeper_classes.generic_implicit_MPI import generic_implicit_MPI

PATH = '/' + os.path.join(*__file__.split('/')[:-1])
SCRIPT = __file__.split('/')[-1].split('.')[0]

COMM_WORLD = MPI.COMM_WORLD

# SDC parameters
nNodes = 4
quadType = 'RADAU-RIGHT'
nodeType = 'LEGENDRE'
parEfficiency = 0.8  # 1/nNodes
nSweeps = 4

# Problem parameters
pName = "ALLEN-CAHN"
tEnd = 50
pParams = {
    "periodic": False,
    "nvars": 2**11 - 1,
    "epsilon": 0.04,
}

# -----------------------------------------------------------------------------
# %% Convergence and error VS cost plots
# -----------------------------------------------------------------------------
nStepsList = np.array([5, 10, 20, 50, 100, 200, 500])
dtVals = tEnd / nStepsList


def getError(uNum, uRef):
    if uNum is None:
        return np.inf
    return np.linalg.norm(uRef[-1, :] - uNum[-1, :], ord=2)


def getCost(counters):
    _, _, tComp = counters
    return tComp


try:
    qDelta = sys.argv[1]
    if qDelta.startswith("--"):
        qDelta = "MIN-SR-FLEX"
except IndexError:
    qDelta = "MIN-SR-FLEX"

try:
    params = getParamsRK(qDelta)
except KeyError:
    params = getParamsSDC(quadType=quadType, numNodes=nNodes, nodeType=nodeType, qDeltaI=qDelta, nSweeps=nSweeps)

useMPI = False
if COMM_WORLD.Get_size() == 4 and qDelta in ["MIN-SR-NS", "MIN-SR-S", "MIN-SR-FLEX", "VDHS"]:  # pragma: no cover
    params['sweeper_class'] = generic_implicit_MPI
    useMPI = True

errors = []
costs = []

root = COMM_WORLD.Get_rank() == 0
if root:
    print(f"Running simulation with {qDelta}")

for nSteps in nStepsList:
    if root:
        uRef = solutionExact(tEnd, nSteps, pName, **pParams)

    uSDC, counters, parallel = solutionSDC(tEnd, nSteps, params, pName, verbose=root, noExcept=True, **pParams)

    if root:
        err = getError(uSDC, uRef)
        errors.append(err)

        cost = getCost(counters)
        costs.append(cost)

if COMM_WORLD.Get_rank() == 0:
    errors = [float(e) for e in errors]

    print("errors : ", errors)
    print("tComps : ", costs)
    fileName = f"{PATH}/fig06_compTime.json"
    timings = {}
    if os.path.isfile(fileName):
        with open(fileName, "r") as f:
            timings = json.load(f)

    timings[qDelta + "_MPI" * useMPI] = {"errors": errors, "costs": costs}

    with open(fileName, 'w') as f:
        json.dump(timings, f, indent=4)
