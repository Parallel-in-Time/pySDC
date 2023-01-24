#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 18:23:16 2023

@author: cpf5546
"""
import numpy as np
from scipy.optimize import newton_krylov
from scipy.integrate import solve_ivp

from pySDC.playgrounds.datatypes.base import DataTypeF
from pySDC.playgrounds.datatypes.implementations import Mesh

# -> data initialization
init = {'shape': (2, 3), 'dtype': float}
u0 = Mesh(**init)
u0.values = [1.0, 0.5, 0.1]

# -> basic check for type preservation
assert type(0.1*u0) == type(u0)
assert type(u0 + 0.1*u0) == type(u0)
assert type(u0[1:]) == type(u0)
assert type(np.exp(u0)) == type(u0)
assert type(u0[None, ...]) == type(u0)
assert type(np.arange(4).reshape((2,2)) @ u0) == type(u0)

# -> RHS evaluation
def evalF(u, t):
    fEval = DataTypeF(u, parts=('impl', 'expl'))
    fEval.impl.values = -2*u + t**2
    fEval.expl.values = 1*u - t
    return fEval

# -> default mass matrix application (must return a DataType instance !)
def applyMassMatrix(u):
    return u  # return 1*u to check DAE limitation in uExact

# -> method for one-step implicit solve : solve (M(u) - factor*f(u,t)) = b
def solveSystem(b, factor, uGuess, t):

    def fun(u):
        u = u0.clone(values=u, copy=False)
        return applyMassMatrix(u) - factor*evalF(u, t) - b

    # Note : can probably add specific default parameters for the newton_krylov solver
    sol = newton_krylov(fun, uGuess)

    return uGuess.clone(values=sol, copy=False)

# -> method for ODE solve
def uExact(t, uInit=None, tInit=0.0):

    uMass = applyMassMatrix(u0)
    if id(uMass) != id(u0):
        raise ValueError('Generic solver for DAE not implemented yet ...')

    uInit = u0 if uInit is None else uInit
    tol = 100 * np.finfo(float).eps

    def fun(t, u):
        u = u0.clone(values=u, copy=False)
        return np.ravel(evalF(u, t).toDataType())

    sol = solve_ivp(fun, (tInit, t), np.ravel(uInit), t_eval=(t,), rtol=tol, atol=tol)

    return u0.clone(values=sol.y, copy=False)

# -> base method to integrate
def integrate(Q, nodes, dt, uNodes):
    out = []
    # integrate RHS over all collocation nodes
    for m in range(len(nodes)):
        # new instance of DataType, initialize values with 0
        out.append(u0.clone(values=0.0))
        for j in range(len(nodes)):
            out[-1] += dt * Q[m, j] * evalF(uNodes[j], nodes[j])

    return out

dt = 0.2
uTh = np.exp(-dt)*(u0-3) + dt**2 - 3*dt + 3
uBE = solveSystem(u0, dt, u0, dt)
uIVP = uExact(dt)
for v in ['uTh', 'uBE', 'uIVP']:
    print(f'{v}:\n{eval(v).__repr__()}')
assert np.allclose(uTh, uIVP)

M = 5
nodes = np.linspace(0, 1, M)
Q = np.ones((M, M))/M
uNodes = [u0.clone(u0.values) for _ in range(M)]
out = integrate(Q, nodes, dt, uNodes)

for u in out:
    assert type(u) == Mesh
