#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick demo to solve the 1D advection-diffusion with Dedalus
using the pySDC interface
"""
# Base user imports
import numpy as np
import matplotlib.pyplot as plt
import dedalus.public as d3

# pySDC imports
from pySDC.playgrounds.dedalus.interface.problem import DedalusProblem
from pySDC.playgrounds.dedalus.interface.sweeper import DedalusSweeperIMEX
from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI

# -----------------------------------------------------------------------------
# User parameters
# -----------------------------------------------------------------------------
listK = [0, 1, 2]   # list of initial wavenumber in the solution (amplitude 1)
nu = 1e-2           # viscosity (unitary velocity)

# -- Space discretisation
xEnd = 2*np.pi      # domain [0, xEnd]
nX = 16             # number of points in x (periodic domain)

# -- Time integration
nSweeps = 4         # number of SDC iterations (sweeps)
nNodes = 4          # number of SDC quadrature nodes
tEnd = 2*np.pi      # final simulation time (starts at 0)
nSteps = 50         # number of time-steps

# -----------------------------------------------------------------------------
# Solver setup
# -----------------------------------------------------------------------------

# -- Dedalus space grid
coords = d3.CartesianCoordinates('x')
dist = d3.Distributor(coords, dtype=np.float64)
xbasis = d3.RealFourier(coords['x'], size=nX, bounds=(0, xEnd))
u = dist.Field(name='u', bases=xbasis)

# -- Initial solution
x = xbasis.local_grid(dist, scale=1)
u0 = np.sum([np.cos(k*x) for k in listK], axis=0)
np.copyto(u['g'], u0)
u0 = u.copy()   # store initial field for later

# -- Problem
dx = lambda f: d3.Differentiate(f, coords['x'])
problem = d3.IVP([u], namespace=locals())
problem.add_equation("dt(u) + dx(u) - nu*dx(dx(u)) = 0")

nSweeps = 4
nNodes = 4
tEnd = 2*np.pi
nSteps = 100
dt = tEnd / nSteps

# -- pySDC controller settings
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

controller = controller_nonMPI(
    num_procs=1, controller_params={'logger_level': 30},
    description=description)

prob = controller.MS[0].levels[0].prob
uSol = prob.solver.state
tVals = np.linspace(0, tEnd, nSteps + 1)

for i in range(nSteps):
    uSol, _ = controller.run(u0=uSol, t0=tVals[i], Tend=tVals[i + 1])

# -- Plotting solution in real and Fourier space
fig, (ax1, ax2) = plt.subplots(1, 2)
fig.suptitle(rf"Advection-diffusion with $\nu={nu:1.1e}$")

plt.sca(ax1)
plt.title("Real space")
plt.plot(u0['g'], label='$u_0$')
plt.plot(u['g'], label='$u(T)$')
plt.legend()
plt.grid(True)
plt.xlabel("$x$")
plt.ylabel("$u$")

plt.sca(ax2)
plt.title("Coefficient space")
plt.plot(u0['c'], 'o', mfc="none", label='$u_0$')
plt.plot(u['c'], 's', mfc="none", ms=10, label='$u(t)$')
plt.legend()
plt.grid(True)
plt.xlabel(r"$\kappa$")
plt.ylabel("$u$")

fig.set_size_inches(12, 5)
plt.tight_layout()
