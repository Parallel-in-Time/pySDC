#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 18:50:39 2023

Utility functions to investigate parallel SDC on non-linear problems
"""
import copy
import numpy as np

from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.implementations.problem_classes.Van_der_Pol_implicit import vanderpol
from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit


def getParamsSDC(quadType="RADAU-RIGHT", numNodes=4, qDeltaI="LU", nSweeps=4):

    description = {
        # Sweeper and its parameters
        "sweeper_class": generic_implicit,
        "sweeper_params": {
            "quad_type": quadType,
            "num_nodes": numNodes,
            "initial_guess": 'spread',
            },
        # Step parameters
        "step_params": {
            "maxiter": nSweeps,
            }
        }

    return description


def setupVanderpol(description, dt=0.1, mu=40):
    """Add Vanderpol settings to pySDC description parameters"""

    # Problem class and parameters
    description["problem_class"] = vanderpol
    description["problem_params"] = {
        'newton_tol': 1e-09,
        'newton_maxiter': 20,
        'mu': mu,   # from 0.1 to 50
        'u0': np.array([2.0, 0]),
        }

    # Level parameters
    description["level_params"] = {
        "restol": 1e-16,
        "dt": dt,
        "nsweeps": 1,
        }


def solVanderpolSDC(t, nSteps, paramsSDC, mu=40):
    dt = t/nSteps
    setupVanderpol(paramsSDC, dt, mu)

    controller = controller_nonMPI(
        num_procs=1,
        controller_params={'logger_level': 30},
        description=paramsSDC
    )

    uinit = controller.MS[0].levels[0].prob.u_exact(0)
    uSDC, _ = controller.run(u0=uinit, t0=0, Tend=t)

    return uSDC


def solVanderpolColl(t, nSteps, paramsSDC, mu=40):
    paramsColl = copy.deepcopy(paramsSDC)
    paramsColl["step_params"]["maxiter"] = 100

    dt = t/nSteps
    setupVanderpol(paramsColl, dt, mu)

    controller = controller_nonMPI(
        num_procs=1,
        controller_params={'logger_level': 30},
        description=paramsColl
    )

    uinit = controller.MS[0].levels[0].prob.u_exact(0)
    uColl, _ = controller.run(u0=uinit, t0=0, Tend=t)

    return uColl


def solVanderpolExact(t, mu=40):
    """Return the exact solution of the Van-der-Pol problem at t"""
    params = getParamsSDC()
    setupVanderpol(params, mu=mu)

    controller = controller_nonMPI(
        num_procs=1,
        controller_params={'logger_level': 30},
        description=params
    )

    return controller.MS[0].levels[0].prob.u_exact(t)
