#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 22 16:05:25 2025

@author: cpf5546
"""
import json
import pandas as pd
import matplotlib.pyplot as plt

from pySDC.playgrounds.dedalus.problems.rbc import checkDNS

simDirs = [
    "run_3D_A4_M0.5_R1_Ra5e3",
    "run_3D_A4_M0.5_R1_Ra1e4",
    "run_3D_A4_M0.5_R1_Ra1.1e4",
    "run_3D_A4_M0.5_R1_Ra1.2e4",
    "run_3D_A4_M0.5_R1_Ra2e4",
    "run_3D_A4_M0.5_R1_Ra1e5",
    ]

simDirs = [
    "run_3D_A4_M1_R1_Ra5e4",
    "run_3D_A4_M1_R1_Ra1e5",
    "run_3D_A4_M1_R1_Ra1.5e5",
    "run_3D_A4_M1_R1_Ra2e5",
    "run_3D_A4_M1_R1_Ra1e6",
    ]

runs = [
    "run_3D_A4_M1_R2_Ra1.5e5",
    "run_3D_A4_M1_R2_Ra3e5",
    "run_3D_A4_M1_R2_Ra5e5",
    "run_3D_A4_M1_R2_Ra8e5",
    "run_3D_A4_M1_R2_Ra9.5e5",
    "run_3D_A4_M1_R2_Ra1e6",
    "run_3D_A4_M1_R2_Ra2e6"
    ]

runs = [
    "run_3D_A4_M1_R4_Ra2e6",
    "run_3D_A4_M1_R4_Ra5e6",
    ]

df = pd.DataFrame(
    columns=["Ra", "c_2[u]", "c_2[uv]", "c_2[uh]", "c_2[b]", "c_2[p]"])

for i, run in enumerate(runs):

    with open(f"postData/{run}.json", "r") as f:
        data = json.load(f)

    df.loc[i, "Ra"] = data["infos"]["Ra"]

    spectrum = data["spectrum"]
    kappa = data["spectrum"]["kappa"]

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

df.columns = ["$"+label+"$" for label in df.columns]
print(df.to_markdown(index=False))
