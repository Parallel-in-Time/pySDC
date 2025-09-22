#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 22 16:05:25 2025

@author: cpf5546
"""
import pandas as pd
import matplotlib.pyplot as plt

from pySDC.playgrounds.dedalus.problems.rbc import OutputFiles, checkDNS

OutputFiles.VERBOSE = True

simDirs = [
    "run_3D_A4_M0.5_R1_Ra5e3",
    "run_3D_A4_M0.5_R1_Ra1e4",
    "run_3D_A4_M0.5_R1_Ra1.1e4",
    "run_3D_A4_M0.5_R1_Ra1.2e4",
    "run_3D_A4_M0.5_R1_Ra2e4",
    "run_3D_A4_M0.5_R1_Ra1e5",
    ]

# simDirs = [
#     "run_3D_A4_M1_R1_Ra5e4",
#     "run_3D_A4_M1_R1_Ra1e5",
#     "run_3D_A4_M1_R1_Ra1.5e5",
#     "run_3D_A4_M1_R1_Ra2e5",
#     "run_3D_A4_M1_R1_Ra1e6",
#     ]

df = pd.DataFrame(
    columns=["Ra", "c_2[u]", "c_2[uv]", "c_2[uh]", "c_2[b]", "c_2[p]"])

for i, dirName in enumerate(simDirs):

    output = OutputFiles(dirName)
    df.loc[i, "Ra"] = float(dirName.split("_Ra")[-1])

    # assert len(output.times) == 61, f"not 61 fields for {dirName}"
    if len(output.times) == 61:
        start = 20
    else:
        start = 60

    spectrum = output.getSpectrum(
            which="all", zVal="all",
            start=start, batchSize=None)
    kappa = output.kappa

    for name in ["u", "uv", "uh", "b", "p"]:
        check = checkDNS(spectrum[name], kappa)
        c2, c1, c0 = check["coeffs"]
        df.loc[i, f"c_2[{name}]"] = float(c2)

plt.figure()
plt.hlines(0, df["Ra"].min(), df["Ra"].max(), linestyles="--", colors="black")
for name, sym in zip(["u", "uv", "uh", "b"], ["s", "^", ">", "o"]):
    plt.semilogx(
        df["Ra"], df[f"c_2[{name}]"],
        sym+"-", label=f"$c_2[{name}]$")
plt.legend()
plt.xlabel("$Ra$")
plt.ylabel("quadratic coefficient $c_2$")
plt.tight_layout()

# df.columns = ["$"+label+"$" for label in df.columns]
