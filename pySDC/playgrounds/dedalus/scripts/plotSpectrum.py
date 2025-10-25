#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 21 14:26:58 2025

@author: cpf5546
"""
import json
import numpy as np
import matplotlib.pyplot as plt


data = [
    ("run_3D_A4_M1_R4_Ra2e6", "$Ra=2e6$"),
    ("run_3D_A4_M1_R4_Ra5e6", "$Ra=5e6$"),
]

plt.figure("spectrum")
for run, label in data:

    with open(f"postData/{run}.json", "r") as f:
        data = json.load(f)

    spectrum = data["spectrum"]
    kappa = np.array(spectrum["kappa"])

    var = "uh"
    plt.loglog(kappa[1:], spectrum[var][1:], label=f"$E({var})$ ({label})")

plt.loglog(kappa[1:], kappa[1:]**(-5/3), '--k')
plt.text(10, 0.1, r"$\kappa^{-5/3}$", fontsize=16)
plt.legend(); plt.grid(True)
plt.ylabel("spectrum"); plt.xlabel(r"wavenumber $\kappa$")
plt.tight_layout()
