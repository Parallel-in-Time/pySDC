#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot string scaling results stored in a given folder
"""
import os
import json
import glob

import numpy as np
import matplotlib.pyplot as plt

# General matplotlib settings
plt.rc('font', size=12)
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16
plt.rcParams['xtick.major.pad'] = 5
plt.rcParams['ytick.major.pad'] = 5
plt.rcParams['axes.labelpad'] = 6
plt.rcParams['markers.fillstyle'] = 'none'
plt.rcParams['lines.markersize'] = 7.0
plt.rcParams['lines.markeredgewidth'] = 1.5
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['mathtext.rm'] = 'serif'
plt.rcParams['figure.max_open_warning'] = 100

COLOR_CYCLE = plt.rcParams["axes.prop_cycle"].by_key()["color"]

folder = (
    "_benchJusuf_64tpc_new"
    # "_benchJureca"
    # "_benchJusuf_64tpc"
    # "_benchJusuf"
)
assert os.path.isdir(folder), f"{folder} is not a folder"

R = 2
# schemes = ["RK443-BCC", "RK443-CCC", "SDC44", "SDC44-PinT-SM-BCC", "SDC44-PinT-SM-CCC", "SDC44-PinT-TM-BCC", "SDC44-PinT-TM-CCC"]
# schemes = ["SDC", "SDC-MPI2-GT"]
schemes = {

    "SDC44": {
        "label": "SDC44", "c": "tab:green", "ls": "x-",
    },
    "SDC44-PinT-SM-CCC": {
        "label": "SDC44 PinT", "c": "tab:green", "ls": "x--"
    },
    "SDC23": {
        "label": "SDC23", "c": "tab:blue", "ls": "o-",
    },
    "SDC23-PinT-SM-CCC": {
        "label": "SDC23 PinT", "c": "tab:blue", "ls": "o--"
    },
    "RK443-BCC": {
        "label": "RK443", "c": "tab:red", "ls": ">-", "ref": True
    },
    "RK111": {
        "label": "RK111", "c": "tab:purple", "ls": "s-"
    },
    # "SDC44-PinT-SM-BCC": {
    #     "label": "SDC44 SM-Block", "c": COLOR_CYCLE[0], "ls": "o-"
    # },
    # "SDC44-PinT-TM-BCC": {
    #     "label": "SDC44 TM-Block", "c": COLOR_CYCLE[0], "ls": "o--"
    # },
    # "SDC44-PinT-TM-CCC": {
    #     "label": "SDC44 TM-Cyclic", "c": COLOR_CYCLE[1], "ls": "x--"
    # }
}

useNSpS = False
nSpS = {
    "RK443": 23,
    "SDC": 17,
    "SDC-MPI": 17,
    "SDC-MPI2": 17,
    "SDC-MPI2-GT": 17,
    }

results = {}

for scheme in schemes.keys():

    files = glob.glob(f"{folder}/R{R}_{scheme}_*.json")
    assert len(files) > 0, f"could not find files for {scheme}"

    results[scheme] = []

    for file in files:

        with open(file, "r") as f:
            infos = json.load(f)

        nAll = infos["Nx"]*infos["Ny"]*infos["Nz"]
        nFactor = nAll*np.log(nAll)
        nFactor = nAll
        nSteps = infos["nSteps"]
        tSim = infos["tComp"]/nSteps/nFactor
        nP = infos["MPI_SIZE"]
        if useNSpS:
            tSim *= nSpS[scheme]
        results[scheme].append([nP, tSim])

    results[scheme].sort(key=lambda p: p[0])

symbols = ["o", "^", "s", "p", "*"]*2
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
ls = "--" if "128tpc" in folder else "-"

label="scaling"+"-nSpS"*useNSpS+f"-R{R}-{folder}"
plt.figure(label)
for scheme, params in schemes.items():
    res = np.array(results[scheme]).T
    plt.loglog(*res, params["ls"], label=params["label"], c=params["c"])
    if "ref" in params:
        plt.loglog(res[0], np.prod(res[:, 0])/res[0], ":", c="gray")
plt.legend()
plt.grid(True)
plt.xlabel("$N_{p}$")
if useNSpS:
    plt.ylabel("$t_{wall}/T_{sim}$")
else:
    plt.ylabel("$t_{wall}/N_{steps}/N_{DoF}$")
plt.tight_layout()
plt.savefig(f"{label}.pdf")


label=f"PinT-efficiency-R{R}-{folder}"
plt.figure(label)


for scheme, params in schemes.items():
    if "PinT" not in scheme:
        continue
    sdcScheme = scheme.split("-")[0]
    nNodes, nSweeps = map(int, sdcScheme[3:])
    spdIdeal = (1+(nSweeps-1)*nNodes)/(1+(nSweeps-1))
    effIdeal = spdIdeal/nNodes
    print(sdcScheme, spdIdeal, effIdeal)
    nProcSpace, tSDC = np.array(results[sdcScheme]).T
    _, tSDCPinT = np.array(results[scheme]).T
    speedup = tSDC[:len(tSDCPinT)]/tSDCPinT
    efficiency = speedup/nNodes
    nPS = nProcSpace[:len(tSDCPinT)]
    plt.semilogx(nPS, efficiency, params["ls"], c=params["c"], label=params["label"])
    # plt.semilogx(nPS, 0*nPS+effIdeal, "--", c="gray")
plt.ylim(0, 1.2)
plt.legend()
plt.grid(True)
plt.xlabel("$N_{p,Space}$"), plt.ylabel("PinT-Efficiency")
plt.tight_layout()
plt.savefig(f"{label}.pdf")
