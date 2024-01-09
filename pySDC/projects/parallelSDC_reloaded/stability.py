#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 10:12:09 2024

Compute stability regions for SDC wit given parameters
"""
import numpy as np
import matplotlib.pyplot as plt

from utils import getParamsRK, getParamsSDC, solutionSDC

SCRIPT = __file__.split('/')[-1].split('.')[0]

# Script parameters
useRK = False
zoom = 2
reLims = -4.5*zoom, 0.5*zoom
imLims = -3.5*zoom, 3.5*zoom
nVals = 251


# RK parameters
rkScheme = "ESDIRK53"

# Collocation parameters
nNodes = 4
nodeType = "LEGENDRE"
quadType = "RADAU-RIGHT"

# SDC parameters
nSweeps = 1
qDeltaType = "MIN-SR-FLEX"


# -----------------------------------------------------------------------------
# Script execution
# -----------------------------------------------------------------------------

# Scheme instanciation
if useRK:
    params = getParamsRK(rkScheme)
else:
    params = getParamsSDC(quadType, nNodes, qDeltaType, nSweeps, nodeType)

# Problem instanciation
reVals = np.linspace(*reLims, num=nVals)
imVals = np.linspace(*imLims, num=nVals)
lambdas = reVals[None, :] + 1j*imVals[:, None]
uNum, counters, parallel = solutionSDC(
    1, 1, params, 'DAHLQUIST', lambdas=lambdas.ravel())

uEnd = uNum[-1, :].reshape(lambdas.shape)
stab = np.abs(uEnd)

fig, axs = plt.subplots(1, 2)

axs[0].contour(reVals, imVals, stab, levels=[1.], colors='black', linewidths=1.5)
axs[0].contourf(reVals, imVals, stab, levels=[1., np.inf], colors='gray')
axs[0].hlines(0, *reLims, linestyles='--', colors='black', linewidth=1)
axs[0].vlines(0, *imLims, linestyles='--', colors='black', linewidth=1)
axs[0].set_aspect('equal', 'box')
axs[0].set_xlabel(r"$Re(\lambda)$")
axs[0].set_ylabel(r"$Im(\lambda)$")
if useRK:
    axs[0].set_title(rkScheme)
else:
    axs[0].set_title(f"{qDeltaType}, K={nSweeps}")

imStab = stab[:, np.argwhere(reVals == 0)].ravel()
axs[1].semilogx(imStab, imVals)
axs[1].tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=True,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=True)
axs[1].vlines(1, *imLims, linestyles='--', colors='black', linewidth=1)
axs[1].set_xlim([0.1, 10])
axs[1].set_ylim(*imLims)
axs[1].set_aspect(0.2)
axs[1].set_xticks([0.1, 1, 10])

plt.tight_layout()
