#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 18:50:39 2023

Utility functions to investigate parallel SDC on non-linear problems
"""
import json
import numpy as np

from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.implementations.problem_classes.Van_der_Pol_implicit import vanderpol, ProblemError
from pySDC.implementations.problem_classes.Lorenz import LorenzAttractor
from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit


def getParamsSDC(
        quadType="RADAU-RIGHT", numNodes=4, qDeltaI="IE", nSweeps=3, dt=0.1):

    description = {
        # Sweeper and its parameters
        "sweeper_class": generic_implicit,
        "sweeper_params": {
            "quad_type": quadType,
            "num_nodes": numNodes,
            "initial_guess": 'spread',
            "QI": qDeltaI,
            },
        # Step parameters
        "step_params": {
            "maxiter": nSweeps,
            },
        # Level parameters
        "level_params": {
            "restol": 1e-16,
            "dt": dt,
            "nsweeps": 1,
            },
        }

    return description


def setupProblem(name, description, **kwargs):
    """Add Vanderpol settings to pySDC description parameters"""

    # Problem class and parameters
    description["problem_class"] = vanderpol
    description["problem_params"] = {
        'newton_tol': 1e-09,
        'newton_maxiter': 300,
        'mu': kwargs.get("mu"),   # vanderpol parameter
        'u0': np.array([2.0, 0]),
        }


def setupLorenz(description, dt=0.1):
    """Add Lorenz settings to pySDC description parameters"""

    # Problem class and parameters
    description["problem_class"] = LorenzAttractor
    description["problem_params"] = {
        'newton_tol': 1e-09,
        'newton_maxiter': 300,
        }

    # Level parameters
    description["level_params"] = {
        "restol": 1e-16,
        "dt": dt,
        "nsweeps": 1,
        }


def solVanderpolSDC(tEnd, nSteps, paramsSDC, mu=10):
    dt = tEnd/nSteps
    setupVanderpol(paramsSDC, dt, mu)

    controller = controller_nonMPI(
        num_procs=1,
        controller_params={'logger_level': 30},
        description=paramsSDC
    )

    prob = controller.MS[0].levels[0].prob

    uInit = prob.u_exact(0)
    uTmp = uInit.copy()

    uSDC = np.zeros((nSteps+1, uInit.size), dtype=uInit.dtype)
    uSDC[0] = uInit

    tVals = np.linspace(0, tEnd, nSteps+1)
    for i in range(nSteps):
        uTmp[:] = uSDC[i]
        try:
            uSDC[i+1], _ = controller.run(u0=uTmp, t0=tVals[i], Tend=tVals[i+1])
        except ProblemError:
            return None, (0, 0)

    nNewton = prob.work_counters["newton"].niter
    nRHS = prob.work_counters["rhs"].niter
    print(f"    done, newton:{nNewton}, rhs:{nRHS}")

    return uSDC, (nNewton, nRHS)


def solVanderpolExact(tEnd, nSteps, mu=10):
    """Return the exact solution of the Van-der-Pol problem at tEnd"""

    key = f"{tEnd}_{nSteps}_{mu}"

    # Eventually load already computed solution from local cache
    try:
        with open('_solVanderpolExact.json', "r") as f:
            cache = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        cache = {}
    if key in cache:
        return np.array(cache[key])

    # Compute solution
    params = getParamsSDC()
    setupVanderpol(params, mu=mu)

    controller = controller_nonMPI(
        num_procs=1,
        controller_params={'logger_level': 30},
        description=params
    )

    tVals = np.linspace(0, tEnd, nSteps+1)
    sol = np.array([controller.MS[0].levels[0].prob.u_exact(t) for t in tVals])

    # Save solution in local cache
    cache[key] = sol.tolist()
    with open('_solVanderpolExact.json', "w") as f:
        json.dump(cache, f)

    return sol
