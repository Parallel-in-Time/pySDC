#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick demo to solve the 1D advection with Dedalus
using the pySDC time-stepper, with convergence plots
"""
import numpy as np
import matplotlib.pyplot as plt
import dedalus.public as d3

from pySDC.playgrounds.dedalus.problems import buildAdvDiffProblem
from pySDC.playgrounds.dedalus.timestepper import SpectralDeferredCorrectionIMEX


# -----------------------------------------------------------------------------
# User parameters
# -----------------------------------------------------------------------------
listK = [0, 1, 2]   # list of initial wavenumber in the solution (amplitude 1)
nX = 128            # number of points in x (periodic domain)

# -----------------------------------------------------------------------------
# Solver setup
# -----------------------------------------------------------------------------
pData = buildAdvDiffProblem(nX, listK)
u = pData["u"]
uInit = pData["u0"]
u0 = uInit['g'].data

# -- Prepare plots
orderPlot = {'RK111': 1,
             'RK222': 2,
             'RK443': 3,
             'ERK4': 4}

SpectralDeferredCorrectionIMEX.setParameters(
    nNodes=4, nodeType='LEGENDRE', quadType='RADAU-RIGHT',
    implSweep='MIN-SR-FLEX', explSweep='PIC', initSweep='COPY')

# -----------------------------------------------------------------------------
# Simulations
# -----------------------------------------------------------------------------
plt.figure('Error')
for timeStepper in [d3.RK111, d3.RK222, d3.RK443, 1, 2, 3]:

    useSDC = False
    nSweeps = None

    if isinstance(timeStepper, int):
        # Using SDC with a given number of sweeps
        nSweeps = timeStepper
        timeStepper = SpectralDeferredCorrectionIMEX
        timeStepper.setParameters(nSweeps=nSweeps)
        useSDC = True

        scheme = f"SDC[{SpectralDeferredCorrectionIMEX.implSweep}, K={nSweeps}]"
    else:
        scheme = timeStepper.__name__
    print(f" -- solving using {scheme}")

    # -- Build solver
    problem = pData["problem"]
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
    lbl = f'SDC, nSweep={nSweeps}' if useSDC else name
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
plt.savefig("demo_timestepper_advDiff_convergence.png")

# Plotting solution in real and Fourier space
fig, (ax1, ax2) = plt.subplots(1, 2)
fig.suptitle("Pure Advection")

plt.sca(ax1)
plt.title("Real space")
plt.plot(uInit['g'], label='$u_0$')
plt.plot(u['g'], '--', label='$u(T)$')
plt.legend()
plt.grid(True)
plt.xlabel("$x$")
plt.ylabel("$u$")

plt.sca(ax2)
plt.title("Coefficient space")
plt.plot(uInit['c'], 'o', mfc="none", label='$u_0$')
plt.plot(u['c'], 's', mfc="none", ms=10, label='$u(t)$')
plt.legend()
plt.grid(True)
plt.xlabel(r"$\kappa$")
plt.ylabel("$u$")

fig.set_size_inches(12, 5)
plt.tight_layout()
plt.savefig("demo_timestepper_advDiff.png")
