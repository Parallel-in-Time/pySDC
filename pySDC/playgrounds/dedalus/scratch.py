#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experiments with dedalus and pySDC
"""
# Base user imports
import numpy as np
import matplotlib.pyplot as plt
import dedalus.public as d3

from problem import DedalusProblem
from sweeper import DedalusSweeperIMEX
from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI


coords = d3.CartesianCoordinates('x')
dist = d3.Distributor(coords, dtype=np.float64)
xbasis = d3.RealFourier(coords['x'], size=16, bounds=(0, 2*np.pi))
u = dist.Field(name='u', bases=xbasis)

# Initial solution
x = xbasis.local_grid(dist, scale=1)
listK = [0, 1, 2]
u0 = np.sum([np.cos(k*x) for k in listK], axis=0)
np.copyto(u['g'], u0)

plt.figure('Initial solution')
plt.plot(u['g'], label='Real space')
plt.plot(u['c'], 'o', label='Coefficient space')
plt.legend()
plt.grid()

# Problem
dx = lambda f: d3.Differentiate(f, coords['x'])
problem = d3.IVP([u], namespace=locals())
problem.add_equation("dt(u) + dx(u) = 0")

nSweeps = 3
nNodes = 4
tEnd = 0.1
nSteps = 1


# pySDC controler settings
description = {
    # Sweeper and its parameters
    "sweeper_class": DedalusSweeperIMEX,
    "sweeper_params": {
        "quad_type": "RADAU-RIGHT",
        "num_nodes": nNodes,
        "node_type": "LEGENDRE",
        "initial_guess": "copy",
        "do_coll_update": False,
        "QI": "IE",
        "QE": "EE",
        'skip_residual_computation': ('IT_CHECK', 'IT_DOWN', 'IT_UP', 'IT_FINE', 'IT_COARSE'),
    },
    # Step parameters
    "step_params": {
        "maxiter": 1,
    },
    # Level parameters
    "level_params": {
        "restol": -1,
        "nsweeps": nSweeps,
    },
    "problem_class": DedalusProblem,
    "problem_params": {
        'problem': problem,
        'nNodes': nNodes,
    }
}

dt = tEnd / nSteps

controller = controller_nonMPI(
    num_procs=1, controller_params={'logger_level': 30},
    description=description)

prob = controller.MS[0].levels[0].prob

uInit = prob.u_exact(0)
uTmp = uInit.copy()

uSDC = np.zeros((nSteps + 1, uInit.size), dtype=uInit.dtype)
uSDC[0] = uInit
tVals = np.linspace(0, tEnd, nSteps + 1)

for i in range(nSteps):
    uTmp[:] = uSDC[i]
    uSDC[i + 1], _ = controller.run(u0=uTmp, t0=tVals[i], Tend=tVals[i + 1])
