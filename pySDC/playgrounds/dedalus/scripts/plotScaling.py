#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot string scaling results stored in a given folder
"""
import json
import glob

import numpy as np
import matplotlib.pyplot as plt

folder = "_benchJureca"

R = 2
if R == 2:
    schemes = ["RK443", "SDC", "SDC-MPI2-GT"] # + ["SDC-MPI", "SDC-MPI2"]
elif R == 1:
    schemes = ["RK443", "SDC", "SDC-MPI2-GT"]

useNSpS = False
nSpS = {
    "RK443": 23,
    "SDC": 17,
    "SDC-MPI": 17,
    "SDC-MPI2": 17,
    "SDC-MPI2-GT": 17,
    }

results = {}

for scheme in schemes:

    files = glob.glob(f"{folder}/R{R}_{scheme}_*.json")

    results[scheme] = []

    for file in files:

        with open(file, "r") as f:
            infos = json.load(f)

        nDOF = 5*infos["Nx"]*infos["Ny"]*infos["Nz"]
        nSteps = infos["nSteps"]
        tSim = infos["tComp"]/nDOF/nSteps
        nP = infos["MPI_SIZE"]
        if useNSpS:
            tSim *= nSpS[scheme]
        results[scheme].append([nP, tSim])

    results[scheme].sort(key=lambda p: p[0])

symbols = ["o", "^", "s", "p", "*"]
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
ls = "--" if "64tpc" in folder else "-"

plt.figure("scaling"+"-nSpS"*useNSpS+f"-R{R}")
for scheme, sym, col in zip(results.keys(), symbols, colors):
    res = np.array(results[scheme]).T
    plt.loglog(*res, sym+ls, label=scheme, c=col)
    # plt.loglog(res[0], np.prod(res[:, 0])/res[0], "--", c="gray")
plt.legend()
plt.grid(True)
plt.xlabel("$N_{p}$")
if useNSpS:
    plt.ylabel("$t_{wall}/N_{DoF}/T_{sim}$")
else:
    plt.ylabel("$t_{wall}/N_{DoF}/N_{steps}$")
plt.tight_layout()


plt.figure(f"PinT-speedup-R{R}")
nProcSpace, tSDC = np.array(results["SDC"]).T
for scheme, sym, col in zip(schemes[2:], symbols, colors):
    _, tSDCPinT = np.array(results[scheme]).T
    speedup = tSDC[:len(tSDCPinT)]/tSDCPinT
    plt.semilogx(nProcSpace[:len(tSDCPinT)], speedup, sym+ls, c=col, label=scheme)
plt.legend()
plt.grid(True)
plt.xlabel("$N_{p,S}$"), plt.ylabel("PinT Speedup")
plt.tight_layout()
