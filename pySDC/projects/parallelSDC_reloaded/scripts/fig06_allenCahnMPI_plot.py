#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 11:14:01 2024

Figures with experiments on the Allen-Cahn problem (MPI runs)
"""
import os
import json
import numpy as np

from pySDC.projects.parallelSDC_reloaded.utils import plt

PATH = '/' + os.path.join(*__file__.split('/')[:-1])
SCRIPT = __file__.split('/')[-1].split('.')[0]

fileName = f"{PATH}/fig06_compTime.json"
timings = {}
with open(fileName, "r") as f:
    timings = json.load(f)

minPrec = ["MIN-SR-NS", "MIN-SR-S", "MIN-SR-FLEX"]

symList = ['^', '>', '<', 'o', 's', '*', 'p']
config = [
    (*minPrec, "VDHS", "ESDIRK43", "LU"),
]
nStepsList = np.array([5, 10, 20, 50, 100, 200, 500])
tEnd = 50
dtVals = tEnd / nStepsList

# -----------------------------------------------------------------------------
# %% Error VS cost plots
# -----------------------------------------------------------------------------
for qDeltaList in config:
    figNameConv = f"{SCRIPT}_conv_1"
    figNameCost = f"{SCRIPT}_cost_1"

    for qDelta, sym in zip(qDeltaList, symList):
        if qDelta == "MIN-SR-NS":
            res = timings["MIN-SR-S_MPI"].copy()
            res["errors"] = [np.nan for _ in res["errors"]]
        elif qDelta in ["MIN-SR-S", "MIN-SR-FLEX", "VDHS"]:
            res = timings[f"{qDelta}_MPI"]
        else:
            res = timings[qDelta]

        errors = res["errors"]
        costs = res["costs"]

        ls = '-' if qDelta.startswith("MIN-SR-") else "--"

        plt.figure(figNameConv)
        plt.loglog(dtVals, errors, sym + ls, label=qDelta)

        plt.figure(figNameCost)
        plt.loglog(costs, errors, sym + ls, label=qDelta)

    for figName in [figNameConv, figNameCost]:
        plt.figure(figName)
        plt.gca().set(
            xlabel="Computation Time [s]" if "cost" in figName else r"$\Delta {t}$",
            ylabel=r"$L_2$ error at $T$",
        )
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{PATH}/{figName}.pdf")

# -----------------------------------------------------------------------------
# %% Speedup tables
# -----------------------------------------------------------------------------
header = f"{'dt size':12} |"
header += '|'.join(f"  {dt:1.1e}  " for dt in dtVals)
print(header)
print("-" * len(header))
meanEff = 0
for qDelta in ["MIN-SR-S", "MIN-SR-FLEX", "VDHS"]:
    seq = timings[f"{qDelta}"]
    par = timings[f"{qDelta}_MPI"]
    assert np.allclose(
        seq["errors"], par["errors"], atol=1e-6
    ), f"parallel and sequential errors are not close for {qDelta}"
    tComp = seq["costs"]
    tCompMPI = par["costs"]
    meanEff += np.mean([tS / tP for tP, tS in zip(tCompMPI, tComp)])
    print(f"{qDelta:12} |" + '|'.join(f" {tS/tP:1.1f} ({tS/tP/4*100:1.0f}%) " for tP, tS in zip(tCompMPI, tComp)))
meanEff /= 3
print("Mean parallel efficiency :", meanEff / 4)
