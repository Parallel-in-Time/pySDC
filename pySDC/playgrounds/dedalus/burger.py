#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Demo script for the KdV-Burgers equation
"""

import numpy as np
import matplotlib.pyplot as plt
import dedalus.public as d3
import logging
logger = logging.getLogger(__name__)

from problem import DedalusProblem
from sweeper import DedalusSweeperIMEX
from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI


# Parameters
Lx = 10
Nx = 1024
a = 1e-4
b = 2e-4
dealias = 3/2
dtype = np.float64

tEnd = 10
nSteps = 10000


# Bases
xcoord = d3.Coordinate('x')
dist = d3.Distributor(xcoord, dtype=dtype)
xbasis = d3.RealFourier(xcoord, size=Nx, bounds=(0, Lx), dealias=dealias)

# Fields
u = dist.Field(name='u', bases=xbasis)

# Substitutions
dx = lambda A: d3.Differentiate(A, xcoord)

# Problem
problem = d3.IVP([u], namespace=locals())
problem.add_equation("dt(u) - a*dx(dx(u)) - b*dx(dx(dx(u))) = - u*dx(u)")

# Initial conditions
x = dist.local_grid(xbasis)
n = 20
u['g'] = np.log(1 + np.cosh(n)**2/np.cosh(n*(x-0.2*Lx))**2) / (2*n)

# pySDC parameters
dt = tEnd / nSteps
nSweeps = 1
nNodes = 4

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
        'skip_residual_computation':
            ('IT_CHECK', 'IT_DOWN', 'IT_UP', 'IT_FINE', 'IT_COARSE'),
    },
    # Step parameters
    "step_params": {
        "maxiter": 1,
    },
    # Level parameters
    "level_params": {
        "dt": dt,
        "restol": -1,
        "nsweeps": nSweeps,
    },
    "problem_class": DedalusProblem,
    "problem_params": {
        'problem': problem,
        'nNodes': nNodes,
    }
}

# Main loop
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
plt.xlim(0, Lx)
plt.ylim(0, tEnd)
plt.xlabel('x')
plt.ylabel('t')
plt.title(f'KdV-Burgers, (a,b)=({a},{b})')
plt.tight_layout()
plt.savefig("KdV_Burgers_pySDC.pdf")
