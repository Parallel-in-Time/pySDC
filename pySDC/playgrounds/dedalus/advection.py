#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Basic script to run the Advection problem with Dedalus
and the SpectralDeferredCorrectionIMEX time integrator
"""
import numpy as np
import dedalus.public as d3
from dedalus_dev import ERK4
from dedalus_dev import SpectralDeferredCorrectionIMEX
from utils import plt  # import matplotlib with improved graph settings

# Bases and field
coords = d3.CartesianCoordinates('x')
dist = d3.Distributor(coords, dtype=np.float64)
xbasis = d3.RealFourier(coords['x'], size=16, bounds=(0, 2*np.pi))
u = dist.Field(name='u', bases=xbasis)

# Initial solution
x = xbasis.local_grid()
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

# Prepare plots
orderPlot = {'RK111': 1,
             'RK222': 2,
             'RK443': 3,
             'ERK4': 4}
plt.figure('Error')

SpectralDeferredCorrectionIMEX.setParameters(
    M=3, quadType='RADAU-RIGHT', nodeDistr='LEGENDRE',
    implSweep='OPT-QmQd-0', explSweep='PIC', initSweep='COPY',
    forceProl=True)

for timeStepper in [d3.RK111, ERK4, 1, 2]:

    # For integer number, use SDC with given number of sweeps
    useSDC = False
    nSweep = None
    if isinstance(timeStepper, int):
        # Using SDC with a given number of sweeps
        nSweep = timeStepper
        timeStepper = SpectralDeferredCorrectionIMEX
        timeStepper.nSweep = nSweep
        useSDC = True

    # Build solver
    solver = problem.build_solver(timeStepper)
    solver.stop_sim_time = 2*np.pi
    name = timeStepper.__name__

    # Function to run the simulation with one given time step
    def getErr(nStep):
        np.copyto(u['g'], u0)
        solver.sim_time = 0
        dt = 2*np.pi/nStep
        for _ in range(nStep):
            solver.step(dt)
        err = np.linalg.norm(u0-u['g'], ord=np.inf)
        return dt, err

    # Run all simulations
    listNumStep = [2**(i+2) for i in range(11)]
    dt, err = np.array([getErr(n) for n in listNumStep]).T

    # Plot error VS time step
    lbl = f'SDC, nSweep={nSweep}' if useSDC else name
    sym = '^-' if useSDC else 'o-'
    plt.loglog(dt, err, sym, label=lbl)

    # Potentially plot order curve
    if name in orderPlot:
        order = orderPlot[name]
        c = err[-1]/dt[-1]**order * 2
        plt.plot(dt, c*dt**order, '--', color='gray')

plt.xlabel(r'$\Delta{t}$')
plt.ylabel(r'error ($L_{inf}$)')
plt.ylim(1e-9, 1e2)
plt.legend()
plt.grid(True)
plt.tight_layout()
