#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Demo script for the KdV-Burgers equation, using the pySDC timestepper
"""
# Base python imports
import numpy as np
import matplotlib.pyplot as plt

import logging
logger = logging.getLogger(__name__)

# pySDC imports
from pySDC.playgrounds.dedalus.problems import buildKdVBurgerProblem
from pySDC.playgrounds.dedalus.timestepper import SDCIMEX

# Dedalus import (for alternative time-stepper)
import dedalus.public as d3

# -----------------------------------------------------------------------------
# User parameters
# -----------------------------------------------------------------------------
xEnd = 10   # space domain size
nX = 512    # number of point in space
nu = 1e-4   # diffusion coefficient
b = 2e-4    # hyper-diffusion coefficient

# -- time integration
tEnd = 10
nSteps = 5000
SDCIMEX.setParameters(
    nSweeps=4,
    nNodes=4,
    implSweep="MIN-SR-FLEX",
    explSweep="PIC")
useSDC = True

# -----------------------------------------------------------------------------
# Solver setup
# -----------------------------------------------------------------------------
timestepper = SDCIMEX if useSDC else d3.RK443
timestep = tEnd/nSteps

pData = buildKdVBurgerProblem(nX, xEnd, nu, b)
problem, u, x = [pData[key] for key in ["problem", "u", "x"]]

solver = problem.build_solver(timestepper)
solver.stop_sim_time = tEnd

# -----------------------------------------------------------------------------
# Simulation run
# -----------------------------------------------------------------------------
u.change_scales(1)
u_list = [np.copy(u['g'])]
t_list = [solver.sim_time]
i = 0
while solver.proceed:
    solver.step(timestep)
    if solver.iteration % 100 == 0:
        print(f"step {solver.iteration}/{nSteps}")
        logger.info('Iteration=%i, Time=%e, dt=%e' %(solver.iteration, solver.sim_time, timestep))
    if solver.iteration % 25 == 0:
        u.change_scales(1)
        u_list.append(np.copy(u['g']))
        t_list.append(solver.sim_time)
    i += 1
solver.log_stats()

# -----------------------------------------------------------------------------
# Plotting solution in real space
# -----------------------------------------------------------------------------
plt.figure(figsize=(6, 4))
plt.pcolormesh(x.ravel(), np.array(t_list), np.array(u_list), cmap='RdBu_r', shading='gouraud', rasterized=True, clim=(-0.8, 0.8))
plt.xlim(0, xEnd)
plt.ylim(0, tEnd)
plt.xlabel('x')
plt.ylabel('t')
plt.title(r'KdV-Burgers, $(\nu,b)='f'({nu},{b})$')
plt.tight_layout()
plt.savefig(f"demo_timestepper_burger{'_SDC' if useSDC else ''}.png")
