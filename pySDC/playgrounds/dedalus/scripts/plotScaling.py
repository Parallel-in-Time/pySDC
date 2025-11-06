#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot string scaling results stored in a given folder
"""
import json
import glob

import numpy as np
import matplotlib.pyplot as plt

folder = "_benchJusuf"

schemes = ["RK443", "SDC", "SDC-MPI", "SDC-MPI2", "SDC-MPI2-GT"]
R = 2

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


plt.figure("scaling"+"-nSpS"*useNSpS)
for scheme, sym in zip(results.keys(), symbols):
    res = np.array(results[scheme]).T
    plt.loglog(*res, sym+'-', label=scheme)
    plt.loglog(res[0], np.prod(res[:, 0])/res[0], "--", c="gray")
plt.legend()
plt.grid(True)
plt.xlabel("$N_{p}$")
if useNSpS:
    plt.ylabel("$t_{wall}/N_{DoF}/T_{sim}$")
else:
    plt.ylabel("$t_{wall}/N_{DoF}/N_{steps}$")
plt.tight_layout()


plt.figure("PinT-speedup")
nProcSpace, tSDC = np.array(results["SDC"]).T
for scheme, sym in zip(schemes[2:], symbols):
    _, tSDCPinT = np.array(results[scheme]).T
    speedup = tSDC[:-2]/tSDCPinT
    plt.semilogx(nProcSpace[:-2], speedup, sym+"-", label=scheme)
plt.legend()
plt.grid(True)
plt.xlabel("$N_{p,S}$"), plt.ylabel("PinT Speedup")
plt.tight_layout()
