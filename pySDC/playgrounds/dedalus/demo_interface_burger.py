#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Demo script for the KdV-Burgers equation
"""
import numpy as np
import matplotlib.pyplot as plt
import logging
logger = logging.getLogger(__name__)

from pySDC.playgrounds.dedalus.problems import buildKdVBurgerProblem
from pySDC.playgrounds.dedalus.interface import DedalusProblem, DedalusSweeperIMEX
from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI

# Space parameters
xEnd = 10
nX = 512
nu = 1e-4
b = 2e-4

# Time-integration parameters
nSweeps = 4
nNodes = 4
tEnd = 10
nSteps = 5000
timeStep = tEnd / nSteps

pData = buildKdVBurgerProblem(nX, xEnd, nu, b)

description = {
    # Sweeper and its parameters
    "sweeper_class": DedalusSweeperIMEX,
    "sweeper_params": {
        "quad_type": "RADAU-RIGHT",
        "num_nodes": nNodes,
        "node_type": "LEGENDRE",
        "initial_guess": "copy",
        "do_coll_update": False,
        "QI": "MIN-SR-S",
        "QE": "PIC",
        'skip_residual_computation':
            ('IT_CHECK', 'IT_DOWN', 'IT_UP', 'IT_FINE', 'IT_COARSE'),
    },
    # Step parameters
    "step_params": {
        "maxiter": 1,
    },
    # Level parameters
    "level_params": {
        "dt": timeStep,
        "restol": -1,
        "nsweeps": nSweeps,
    },
    "problem_class": DedalusProblem,
    "problem_params": {
        'problem': pData["problem"],
        'nNodes': nNodes,
    }
}

# Main loop
u, x = [pData[key] for key in ["u", "x"]]
u.change_scales(1)
u_list = [np.copy(u['g'])]
t_list = [0]

size = u_list[0].size

controller = controller_nonMPI(
    num_procs=1, controller_params={'logger_level': 30},
    description=description)

prob = controller.MS[0].levels[0].prob
uSol = prob.solver.state

tVals = np.linspace(0, tEnd, nSteps + 1)

for i in range(nSteps):
    uSol, _ = controller.run(u0=uSol, t0=tVals[i], Tend=tVals[i + 1])
    if (i+1) % 100 == 0:
        print(f"step {i+1}/{nSteps}")
    if (i+1) % 25 == 0:
        u.change_scales(1)
        u_list.append(np.copy(u['g']))
        t_list.append(tVals[i])


# Plot
plt.figure(figsize=(6, 4))
plt.pcolormesh(
    x.ravel(), np.array(t_list), np.array(u_list), cmap='RdBu_r',
    shading='gouraud', rasterized=True, clim=(-0.8, 0.8))
plt.xlim(0, xEnd)
plt.ylim(0, tEnd)
plt.xlabel('x')
plt.ylabel('t')
plt.title(r'KdV-Burgers, $(\nu,b)='f'({nu},{b})$')
plt.tight_layout()
plt.savefig("demo_interface_burger.png")
