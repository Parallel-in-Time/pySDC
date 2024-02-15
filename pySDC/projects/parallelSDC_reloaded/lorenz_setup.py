#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 10:08:04 2023

Script to numerically compute number revolution periods for the Lorenz system
"""
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

from pySDC.projects.parallelSDC_reloaded.utils import solutionExact

script = __file__.split('/')[-1].split('.')[0]

tEnd = 2
nSteps = tEnd * 50
tVals = np.linspace(0, tEnd, nSteps + 1)

nPeriods = 2

print(f"Computing exact solution up to t={tEnd} ...")
uExact = solutionExact(tEnd, nSteps, "LORENZ", u0=(5, -5, 20))

z = uExact[:, -1]
idx = signal.find_peaks(z)[0][nPeriods - 1]


print(f'tEnd for {nPeriods} periods : {tVals[idx]}')

figName = f"{script}_traj"

plt.figure(figName)
plt.plot(tVals, uExact[:, 0], '-', label="$x(t)$")
plt.plot(tVals, uExact[:, 1], '-', label="$y(t)$")
plt.plot(tVals, uExact[:, 2], '-', label="$z(t)$")
plt.vlines(tVals[idx], ymin=-20, ymax=40, linestyles="--", linewidth=1)

plt.legend()
plt.xlabel("time")
plt.ylabel("trajectory")
plt.tight_layout()
