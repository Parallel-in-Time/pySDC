#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 21:35:05 2023

Script to numerically determine periods of Van der Pol oscillation for
different mu parameters.
"""
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

from pySDC.projects.parallelSDC_reloaded.utils import solutionExact

script = __file__.split('/')[-1].split('.')[0]


muVals = [0.1, 2, 10]
muPeriods = []

tEnd = 20
nSteps = 200
tVals = np.linspace(0, tEnd, nSteps + 1)

# Compute and plot unscaled solution to determined period for each mu
for mu in muVals:
    print(f"Computing exact solution up to t={tEnd} for mu={mu} ...")
    uExact = solutionExact(tEnd, nSteps, "VANDERPOL", mu=mu)
    plt.figure(f"{script}_traj")
    plt.plot(tVals, uExact[:, 0], '-', label=f"$\\mu={mu}$")
    plt.figure(f"{script}_accel")
    plt.plot(tVals, uExact[:, 1], '-', label=f"$\\mu={mu}$")

    x = uExact[:, 0]
    idx = signal.find_peaks(x)[0][0]
    period = tVals[idx]
    print(f" -- done, found period={period:.1f}")
    muPeriods.append(period)

# Compute and plot solution for each mu on one period, scale time with period
for mu, tEnd in zip(muVals, muPeriods):
    nSteps = 200
    tVals = np.linspace(0, tEnd, nSteps + 1)

    print(f"Computing exact solution up to t={tEnd:.1f} for mu={mu} ...")
    uExact = solutionExact(tEnd, nSteps, "VANDERPOL", mu=mu)
    plt.figure(f"{script}_traj_scaled")
    plt.plot(tVals / tEnd, uExact[:, 0], '-', label=f"$\\mu={mu}$")
    print(' -- done')

# Figure settings
for figName in [f"{script}_traj", f"{script}_accel", f"{script}_traj_scaled"]:
    plt.figure(figName)
    plt.legend()
    plt.xlabel("time (scaled)" if "scaled" in figName else "time")
    plt.ylabel("trajectory" if "traj" in figName else "acceleration")
    plt.tight_layout()
