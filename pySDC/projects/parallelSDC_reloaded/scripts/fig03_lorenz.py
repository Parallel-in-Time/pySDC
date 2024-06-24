#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 15:34:24 2024

Figures with experiment on the Lorenz problem
"""
import os
import numpy as np
import scipy as sp

from pySDC.projects.parallelSDC_reloaded.utils import solutionExact, getParamsSDC, solutionSDC, getParamsRK, plt
from pySDC.helpers.testing import DataChecker

data = DataChecker(__file__)

PATH = '/' + os.path.join(*__file__.split('/')[:-1])
SCRIPT = __file__.split('/')[-1].split('.')[0]

symList = ['o', '^', 's', '>', '*', '<', 'p', '>'] * 10

# SDC parameters
nNodes = 4
quadType = 'RADAU-RIGHT'
nodeType = 'LEGENDRE'
parEfficiency = 0.8  # 1/nNodes

# -----------------------------------------------------------------------------
# Trajectories (reference solution)
# -----------------------------------------------------------------------------
tEnd = 2
nSteps = tEnd * 50
tVals = np.linspace(0, tEnd, nSteps + 1)
nPeriods = 2

print(f"Computing exact solution up to t={tEnd} ...")
uExact = solutionExact(tEnd, nSteps, "LORENZ", u0=(5, -5, 20))

z = uExact[:, -1]
idx = sp.signal.find_peaks(z)[0][nPeriods - 1]
print(f'tEnd for {nPeriods} periods : {tVals[idx]}')

figName = f"{SCRIPT}_traj"
plt.figure(figName)
me = 0.1
plt.plot(tVals, uExact[:, 0], 's-', label="$x(t)$", markevery=me)
plt.plot(tVals, uExact[:, 1], 'o-', label="$y(t)$", markevery=me)
plt.plot(tVals, uExact[:, 2], '^-', label="$z(t)$", markevery=me)
plt.vlines(tVals[idx], ymin=-20, ymax=40, linestyles="--", linewidth=1)
plt.legend(loc="upper right")
plt.xlabel("$t$")
plt.ylabel("Trajectory")
plt.gcf().set_size_inches(12, 3)
plt.tight_layout()
plt.savefig(f'{PATH}/{figName}.pdf')

# -----------------------------------------------------------------------------
# %% Convergence plots
# -----------------------------------------------------------------------------
tEnd = 1.24
nStepsList = np.array([2, 5, 10, 20, 50, 100, 200, 500, 1000])
dtVals = tEnd / nStepsList


def getError(uNum, uRef):
    if uNum is None:  # pragma: no cover
        return np.inf
    return np.linalg.norm(np.linalg.norm(uRef - uNum, np.inf, axis=-1), np.inf)


config = ["PIC", "MIN-SR-NS"]
for qDelta, sym in zip(config, symList):
    figName = f"{SCRIPT}_conv_{qDelta}"
    plt.figure(figName)

    for nSweeps in [1, 2, 3, 4, 5]:
        params = getParamsSDC(quadType=quadType, numNodes=nNodes, nodeType=nodeType, qDeltaI=qDelta, nSweeps=nSweeps)

        errors = []

        for nSteps in nStepsList:
            print(f' -- nSteps={nSteps} ...')

            uRef = solutionExact(tEnd, nSteps, "LORENZ", u0=(5, -5, 20))

            uSDC, counters, parallel = solutionSDC(tEnd, nSteps, params, "LORENZ", u0=(5, -5, 20))

            err = getError(uSDC, uRef)
            errors.append(err)

        # error VS dt
        label = f"$K={nSweeps}$"
        plt.loglog(dtVals, errors, sym + '-', label=f"$K={nSweeps}$")
        data.storeAndCheck(f"{figName}_{label}", errors[1:])

    x = dtVals[4:]
    for k in [1, 2, 3, 4, 5, 6]:
        plt.loglog(x, 1e4 * x**k, "--", color="gray", linewidth=0.8)

    plt.gca().set(
        xlabel=r"$\Delta{t}$",
        ylabel=r"$L_\infty$ error",
        ylim=(8.530627786509715e-12, 372.2781393394293),
    )
    plt.legend(loc="lower right")
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"{PATH}/{figName}.pdf")


# -----------------------------------------------------------------------------
# %% Error VS cost plots
# -----------------------------------------------------------------------------
def getCost(counters):
    nNewton, nRHS, tComp = counters
    return nNewton + nRHS


minPrec = ["MIN-SR-NS", "MIN-SR-S", "MIN-SR-FLEX"]

symList = ['^', '>', '<', 'o', 's', '*']
config = [
    [(*minPrec, "LU", "EE", "PIC"), 4],
    [(*minPrec, "VDHS", "RK4", "ESDIRK43"), 4],
    [(*minPrec, "PIC", "RK4", "ESDIRK43"), 5],
]


i = 0
for qDeltaList, nSweeps in config:
    figName = f"{SCRIPT}_cost_{i}"
    i += 1
    plt.figure(figName)

    for qDelta, sym in zip(qDeltaList, symList):
        try:
            params = getParamsRK(qDelta)
        except KeyError:
            params = getParamsSDC(
                quadType=quadType, numNodes=nNodes, nodeType=nodeType, qDeltaI=qDelta, nSweeps=nSweeps
            )

        errors = []
        costs = []

        for nSteps in nStepsList:
            uRef = solutionExact(tEnd, nSteps, "LORENZ", u0=(5, -5, 20))

            uSDC, counters, parallel = solutionSDC(tEnd, nSteps, params, "LORENZ", u0=(5, -5, 20))

            err = getError(uSDC, uRef)
            errors.append(err)

            cost = getCost(counters)
            if parallel:
                assert qDelta != "EE", "wait, whaaat ??"
                cost /= nNodes * parEfficiency
            costs.append(cost)

        # error VS cost
        ls = '-' if qDelta.startswith("MIN-SR-") else "--"
        plt.loglog(costs, errors, sym + ls, label=qDelta)
        data.storeAndCheck(f"{figName}_{qDelta}", errors[2:], rtol=1e-2)

    plt.gca().set(
        xlabel="Cost",
        ylabel=r"$L_\infty$ error",
        ylim=(1e-10, 400),
        xlim=(30, 20000),
    )
    plt.legend(loc="lower left")
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"{PATH}/{figName}.pdf")

data.writeToJSON()
