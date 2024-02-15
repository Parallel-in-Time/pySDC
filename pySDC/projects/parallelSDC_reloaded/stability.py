#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 10:12:09 2024

Compute stability regions for SDC wit given parameters
"""
import numpy as np
from pySDC.projects.parallelSDC_reloaded.utils import getParamsRK, getParamsSDC, solutionSDC, plotStabContour, plt

SCRIPT = __file__.split('/')[-1].split('.')[0]

# Script parameters
useRK = False
zoom = 20
reLims = -4.5 * zoom, 0.5 * zoom
imLims = -3.5 * zoom, 3.5 * zoom
nVals = 251


# RK parameters
rkScheme = "RK4"

# Collocation parameters
nNodes = 4
nodeType = "LEGENDRE"
quadType = "RADAU-RIGHT"

# SDC parameters
nSweeps = 6
qDeltaType = "VDHS"
collUpdate = False


# -----------------------------------------------------------------------------
# Script execution
# -----------------------------------------------------------------------------

# Scheme instanciation
if useRK:  # pragma: no cover
    params = getParamsRK(rkScheme)
else:
    params = getParamsSDC(quadType, nNodes, qDeltaType, nSweeps, nodeType, collUpdate)

# Problem instanciation
reVals = np.linspace(*reLims, num=nVals)
imVals = np.linspace(*imLims, num=nVals)
lambdas = reVals[None, :] + 1j * imVals[:, None]
uNum, counters, parallel = solutionSDC(1, 1, params, 'DAHLQUIST', lambdas=lambdas.ravel())

uEnd = uNum[-1, :].reshape(lambdas.shape)
stab = np.abs(uEnd)

fig, axs = plt.subplots(1, 2)

ax = plotStabContour(reVals, imVals, stab, ax=axs[0])
if useRK:  # pragma: no cover
    ax.set_title(rkScheme)
else:
    ax.set_title(f"{qDeltaType}, K={nSweeps}")

imStab = stab[:, np.argwhere(reVals == 0)].ravel()
axs[1].semilogx(imStab, imVals)
axs[1].tick_params(
    axis='x',  # changes apply to the x-axis
    which='both',  # both major and minor ticks are affected
    bottom=True,  # ticks along the bottom edge are off
    top=False,  # ticks along the top edge are off
    labelbottom=True,
)
axs[1].vlines(1, *imLims, linestyles='--', colors='black', linewidth=1)
axs[1].set_xlim([0.1, 10])
axs[1].set_ylim(*imLims)
axs[1].set_aspect(0.2)
axs[1].set_xticks([0.1, 1, 10])
axs[1].set_title("Imaginary axis")

plt.tight_layout()
