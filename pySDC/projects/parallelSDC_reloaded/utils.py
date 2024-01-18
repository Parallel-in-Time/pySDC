#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 18:50:39 2023

Utility functions to investigate parallel SDC on non-linear problems
"""
import json
import numpy as np
from time import time
import traceback

from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI

from pySDC.implementations.problem_classes.Van_der_Pol_implicit import vanderpol
from pySDC.implementations.problem_classes.Lorenz import LorenzAttractor
from pySDC.implementations.problem_classes.odeScalar import ProtheroRobinson
from pySDC.implementations.problem_classes.odeSystem import Kaps, ChemicalReaction3Var, JacobiElliptic
from pySDC.implementations.problem_classes.AllenCahn_1D_FD import allencahn_front_fullyimplicit, allencahn_periodic_fullyimplicit
from pySDC.implementations.problem_classes.TestEquation_0D import testequation0d

from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit
import pySDC.implementations.sweeper_classes.Runge_Kutta as rk

import matplotlib.pyplot as plt

# General matplotlib settings
plt.rc('font', size=12)
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['xtick.labelsize'] = 15
plt.rcParams['ytick.labelsize'] = 15
plt.rcParams['xtick.major.pad'] = 3
plt.rcParams['ytick.major.pad'] = 2
plt.rcParams['axes.labelpad'] = 6
plt.rcParams['markers.fillstyle'] = 'none'
plt.rcParams['lines.markersize'] = 7.0
plt.rcParams['lines.markeredgewidth'] = 1.5
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['mathtext.rm'] = 'serif'
plt.rcParams['figure.max_open_warning'] = 100

def getParamsSDC(
        quadType="RADAU-RIGHT", numNodes=4, qDeltaI="IE", nSweeps=3,
        nodeType="LEGENDRE", collUpdate=False):

    description = {
        # Sweeper and its parameters
        "sweeper_class": generic_implicit,
        "sweeper_params": {
            "quad_type": quadType,
            "num_nodes": numNodes,
            "node_type": nodeType,
            "initial_guess": 'spread',
            "do_coll_update": collUpdate,
            "QI": qDeltaI,
            'skip_residual_computation':
                ('IT_CHECK', 'IT_DOWN', 'IT_UP', 'IT_FINE', 'IT_COARSE')
            },
        # Step parameters
        "step_params": {
            "maxiter": 1,
            },
        # Level parameters
        "level_params": {
            "restol": -1,
            "nsweeps": nSweeps,
            }
        }

    return description


RK_SWEEPERS = {
    "BE": rk.BackwardEuler,
    "FE": rk.ForwardEuler,
    "RK4": rk.RK4,
    "DIRK43": rk.DIRK43,
    "ESDIRK53": rk.ESDIRK53,
    "ESDIRK43": rk.ESDIRK43,
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
        # Level parameters
        "level_params": {
            "restol": -1,
            "nsweeps": 1,
            }
        }

    return description


def setupProblem(name, description, dt, **kwargs):
    """Add problem settings to pySDC description parameters"""

    # Common newton tolerance and max number of iterations
    description["problem_params"] = {
        'newton_tol': 1e-13,
        'newton_maxiter': 300,
        }
    # Level parameters
    description["level_params"]["dt"] = dt

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
    elif name == "KAPS":
        description["problem_class"] = Kaps
        description["problem_params"].update({
            'epsilon': kwargs.get("epsilon", 1e-3),
            })
    elif name == "CHEMREC":
        description["problem_class"] = ChemicalReaction3Var
    elif name == "ALLEN-CAHN":
        periodic = kwargs.get("periodic", False)
        description["problem_class"] = allencahn_periodic_fullyimplicit if periodic \
            else allencahn_front_fullyimplicit
        description["problem_params"].update({
            'nvars': kwargs.get("nvars", 128 if periodic else 127),
            'eps': kwargs.get("epsilon", 0.04),
            })
    elif name == "JACELL":
        description["problem_class"] = JacobiElliptic
    elif name == "DAHLQUIST":
        lambdas = kwargs.get("lambdas", None)
        description["problem_class"] = testequation0d
        description["problem_params"].update({
            "lambdas": lambdas,
            "u0": 1.0,
            })
        description["problem_params"].pop("newton_tol")
        description["problem_params"].pop("newton_maxiter")
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
    print(" -- computing numerical solution with pySDC ...")
    for i in range(nSteps):
        uTmp[:] = uSDC[i]
        try:
            uSDC[i+1], _ = controller.run(u0=uTmp, t0=tVals[i], Tend=tVals[i+1])
        except Exception as e:
            print(f" -- exception when running controller : {e}")
            traceback.print_exc()
            return None, (0, 0, 0), False
    tComp = time() - tBeg

    try:
        nNewton = prob.work_counters["newton"].niter
    except KeyError:
        nNewton = 0
    nRHS = prob.work_counters["rhs"].niter
    print(f"    done, newton:{nNewton}, rhs:{nRHS}, tComp:{tComp}")
    try:
        parallel = controller.MS[0].levels[0].sweep.parallelizable
    except AttributeError:
        parallel = False

    return uSDC, (nNewton, nRHS, tComp), parallel


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
    elif probName == "CHEMREC":
        key = f"{tEnd}_{nSteps}"
        cacheFile = '_solChemicalReactionExact.json'
    elif probName == "JACELL":
        key = f"{tEnd}_{nSteps}"
        cacheFile = '_solJacobiEllipticExact.json'

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

    print(" -- computing analytical solution with P.u_exact ...")
    tBeg = time()
    tVals = np.linspace(0, tEnd, nSteps+1)
    uExact = [solver(0)]
    for i in range(nSteps):
        uExact.append(solver(tVals[i+1], uExact[-1], tVals[i]))
    uExact = np.array(uExact)
    print(f"    done in {time()-tBeg:1.2f}s")

    try:
        # Save solution in local cache
        cache[key] = uExact.tolist()
        with open(cacheFile, "w") as f:
            json.dump(cache, f)
    except UnboundLocalError:
        pass

    return uExact


# Plotting functions
def plotStabContour(reVals, imVals, stab, ax=None):
    if ax is None: ax=plt.gca()
    ax.contour(reVals, imVals, stab, levels=[1.], colors='black', linewidths=1)
    ax.contourf(reVals, imVals, stab, levels=[1., np.inf], colors='gainsboro')
    ax.hlines(0, min(reVals), max(reVals), linestyles='--', colors='black', linewidth=0.5)
    ax.vlines(0, min(imVals), max(imVals), linestyles='--', colors='black', linewidth=0.5)
    ax.set_aspect('equal', 'box')
    ax.set_xlabel(r"$Re(\lambda)$", labelpad=0, fontsize=10)
    ax.set_ylabel(r"$Im(\lambda)$", labelpad=0, fontsize=10)
    return ax
