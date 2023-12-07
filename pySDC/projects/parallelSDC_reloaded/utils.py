#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 18:50:39 2023

Utility functions to investigate parallel SDC on non-linear problems
"""
import json
import numpy as np
from time import time

from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.implementations.problem_classes.Van_der_Pol_implicit import vanderpol
from pySDC.implementations.problem_classes.Lorenz import LorenzAttractor
from pySDC.implementations.problem_classes.odeScalar import ProtheroRobinson
from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit
import pySDC.implementations.sweeper_classes.Runge_Kutta as rk

import matplotlib.pyplot as _plt

# General matplotlib settings
_plt.rc('font', size=12)
_plt.rcParams['lines.linewidth'] = 2
_plt.rcParams['axes.titlesize'] = 18
_plt.rcParams['axes.labelsize'] = 16
_plt.rcParams['xtick.labelsize'] = 16
_plt.rcParams['ytick.labelsize'] = 16
_plt.rcParams['xtick.major.pad'] = 5
_plt.rcParams['ytick.major.pad'] = 5
_plt.rcParams['axes.labelpad'] = 6
_plt.rcParams['markers.fillstyle'] = 'none'
_plt.rcParams['lines.markersize'] = 7.0
_plt.rcParams['lines.markeredgewidth'] = 1.5
_plt.rcParams['mathtext.fontset'] = 'cm'
_plt.rcParams['mathtext.rm'] = 'serif'
_plt.rcParams['figure.max_open_warning'] = 100

def getParamsSDC(
        quadType="RADAU-RIGHT", numNodes=4, qDeltaI="IE", nSweeps=3,
        nodeType="LEGENDRE"):

    description = {
        # Sweeper and its parameters
        "sweeper_class": generic_implicit,
        "sweeper_params": {
            "quad_type": quadType,
            "num_nodes": numNodes,
            "node_type": nodeType,
            "initial_guess": 'spread',
            "QI": qDeltaI,
            'skip_residual_computation':
                ('IT_CHECK', 'IT_DOWN', 'IT_UP', 'IT_FINE', 'IT_COARSE')
            },
        # Step parameters
        "step_params": {
            "maxiter": nSweeps,
            },
        }

    return description


RK_SWEEPERS = {
    "RK4": rk.RK4,
    "DIRK43": rk.DIRK43,
    "ESDIRK53": rk.ESDIRK53
}


def getParamsRK(method="RK4"):

    description = {
        # Sweeper and its parameters
        "sweeper_class": RK_SWEEPERS[method],
        "sweeper_params": {
            'skip_residual_computation':
                ('IT_CHECK', 'IT_DOWN', 'IT_UP', 'IT_FINE', 'IT_COARSE')
            },
        # Step parameters
        "step_params": {
            "maxiter": 1,
            },
        }

    return description


def setupProblem(name, description, dt, **kwargs):
    """Add problem settings to pySDC description parameters"""

    # Common newton tolerance and max number of iterations
    description["problem_params"] = {
        'newton_tol': 1e-08,
        'newton_maxiter': 300,
        }
        # Level parameters
    description["level_params"] = {
        "restol": -1,
        "dt": dt,
        "nsweeps": 1,
        }

    if name == "VANDERPOL":
        description["problem_class"] = vanderpol
        description["problem_params"].update({
            'mu': kwargs.get("mu", 10),   # vanderpol parameter
            'u0': np.array([2.0, 0]),
            })
    elif name == "LORENZ":
        description["problem_class"] = LorenzAttractor
        description["problem_params"].update({
            'u0': kwargs.get("u0", (1, 1, 1)),
            })
    elif name == "PROTHERO-ROBINSON":
        description["problem_class"] = ProtheroRobinson
        description["problem_params"].update({
            'epsilon': kwargs.get("epsilon", 1e-3),
            })
    elif name == "PROTHERO-ROBINSON-NL":
        description["problem_class"] = ProtheroRobinson
        description["problem_params"].update({
            'epsilon': kwargs.get("epsilon", 1e-3),
            'nonLinear': True,
            })
    else:
        raise NotImplementedError(f"problem {name} not implemented")


def solutionSDC(tEnd, nSteps, params, probName, **kwargs):
    dt = tEnd/nSteps
    setupProblem(probName, params, dt, **kwargs)

    controller = controller_nonMPI(
        num_procs=1,
        controller_params={'logger_level': 30},
        description=params
    )

    prob = controller.MS[0].levels[0].prob

    uInit = prob.u_exact(0)
    uTmp = uInit.copy()

    uSDC = np.zeros((nSteps+1, uInit.size), dtype=uInit.dtype)
    uSDC[0] = uInit

    tVals = np.linspace(0, tEnd, nSteps+1)
    tBeg = time()
    for i in range(nSteps):
        uTmp[:] = uSDC[i]
        try:
            uSDC[i+1], _ = controller.run(u0=uTmp, t0=tVals[i], Tend=tVals[i+1])
        except Exception:
            return None, (0, 0, 0)
    tComp = time() - tBeg

    nNewton = prob.work_counters["newton"].niter
    nRHS = prob.work_counters["rhs"].niter
    print(f"    done, newton:{nNewton}, rhs:{nRHS}, tComp:{tComp}")

    return uSDC, (nNewton, nRHS, tComp)


def solutionExact(tEnd, nSteps, probName, **kwargs):
    """Return the exact solution of the Van-der-Pol problem at tEnd"""

    if probName == "VANDERPOL":
        mu = kwargs.get('mu', 10)
        key = f"{tEnd}_{nSteps}_{mu}"
        cacheFile = '_solVanderpolExact.json'
    elif probName == "LORENZ":
        u0 = kwargs.get('u0', (1, 1, 1))
        key = f"{tEnd}_{nSteps}_{u0}"
        cacheFile = '_solLorenzExact.json'



    # Eventually load already computed solution from local cache
    try:
        with open(cacheFile, "r") as f:
            cache = json.load(f)
        if key in cache:
            return np.array(cache[key])
    except (FileNotFoundError, json.JSONDecodeError, UnboundLocalError):
        cache = {}

    # Compute solution
    params = getParamsSDC()
    dt = tEnd/nSteps
    setupProblem(probName, params, dt, **kwargs)

    controller = controller_nonMPI(
        num_procs=1,
        controller_params={'logger_level': 30},
        description=params
    )
    solver = controller.MS[0].levels[0].prob.u_exact

    tVals = np.linspace(0, tEnd, nSteps+1)
    uExact = [solver(0)]
    for i in range(nSteps):
        uExact.append(solver(tVals[i+1], uExact[-1], tVals[i]))
    uExact = np.array(uExact)

    try:
        # Save solution in local cache
        cache[key] = uExact.tolist()
        with open(cacheFile, "w") as f:
            json.dump(cache, f)
    except UnboundLocalError:
        pass

    return uExact


def solutionRK(tEnd, nSteps, probName, method="RK4"):
    pass
