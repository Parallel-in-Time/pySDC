#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 18:16:35 2023

@author: cpf5546
"""
import numpy as np
from petsc4py import PETSc

from scipy.optimize import newton_krylov
from scipy.integrate import solve_ivp

# -----------------------------------------------------------------------------
# Base core implementation
# -----------------------------------------------------------------------------
try:
    from mpi4py import MPI
except ImportError:
    MPI = None


class DataType(object):
    """Base datatype class for solution"""

    # Class attribute, allowed to be changed during run-time for all instantiated datatypes
    _comm = None if MPI is None else MPI.COMM_WORLD

    def isend(self, dest=None, tag=None):
        return self._comm.Issend(self.values, dest=dest, tag=tag)

    def irecv(self, source=None, tag=None):
        return self._comm.Irecv(self.values, source=source, tag=tag)

    def bcast(self, root=None):
        self._comm.Bcast(self.values, root=root)
        return self

    @property
    def values(self):
        # Must be redefined in children class
        raise NotImplementedError()

    @values.setter
    def values(self, values):
        # Must be redefined in children class
        raise NotImplementedError()

    @property
    def clone(self, values=None, copy=True):
        # Must be redefined in children class
        raise NotImplementedError()


class DataTypeF(object):
    """Base datatype class for f(u,t) evaluations"""

    def __new__(cls, dataType, parts=()):
        if len(parts) == 0:
            return dataType.clone()
        else:
            obj = super().__new__(cls)
            for name in parts:
                super().__setattr__(obj, name, dataType.clone())
            super().__setattr__(obj, '_parts', set(parts))
            obj._dataType = dataType
            return obj

    @property
    def partNames(self):
        return self._parts

    @property
    def parts(self):
        return [self.__getattribute__(name) for name in self._parts]

    def __setattr__(self, name, value):
        if name in self._parts:
            raise ValueError(f'{name} is read-only')
        super().__setattr__(name, value)

    def __repr__(self):
        return '{' + ',\n '.join(
            f'{c}: {getattr(self, c).__repr__()}' for c in self._parts) + '}'

    def toDataType(self):
        out = self._dataType.clone(values=self.parts[0].values)
        for c in self.parts[1:]:
            out.values += c.values
        return out

    def __add__(self, other):
        if type(other) == DataTypeF:
            raise ValueError()
        out = self.toDataType()
        out += other
        return out

    def __mul__(self, other):
        if type(other) == DataTypeF:
            raise ValueError()
        out = self.toDataType()
        out *= other
        return out

    def __neg__(self):
        out = self.toDataType()
        out *= -1
        return out

    def __sub__(self, other):
        if type(other) == DataTypeF:
            raise ValueError()
        out = self.toDataType()
        out -= other
        return out

    def __radd__(self, other):
        return self.__add__(other)

    def __rmul__(self, other):
        return self.__mul__(other)


# -----------------------------------------------------------------------------
# Specific implementations (in pySDC/implementations)
# -----------------------------------------------------------------------------
class Mesh(DataType, np.ndarray):
    # Note : priority to methods from DataType class

    # Mandatory redefinitions
    @property
    def values(self):
        return self[:]  # maybe 'self' alone is sufficient ...

    @values.setter
    def values(self, values):
        np.copyto(self, values)

    def clone(self, values=None, copy=True):
        if values is None:
            return Mesh(self.shape, self.dtype)
        if not copy:
            return np.asarray(values).view(Mesh).reshape(self.shape)
        out = Mesh(self.shape, self.dtype)
        out.values = values
        return out

    # Additional redefinitions
    def __abs__(self):
        local_absval = float(np.amax(np.ndarray.__abs__(self)))

        if (self._comm is not None) and (self._comm.Get_size() > 1):
            global_absval = 0.0
            global_absval = max(self._comm.allreduce(sendobj=local_absval, op=MPI.MAX), global_absval)
        else:
            global_absval = local_absval
        return float(global_absval)

class PETScMesh(DataType, PETSc.Vec):
    # Note : priority to methods from DataType class

    # Mandatory redefinitions
    @property
    def values(self):
        return self.getArray()

    @values.setter
    def values(self, values):
        np.copyto(self.getArray(), values)

    def clone(self, values=None):
        # TODO : implementation
        pass

    # Additional redefinitions
    def __abs__(self):
        return self.norm(3)


class Particles(Mesh):
    # TODO : understand what p and q are in the original implementation

    def __init__(self, nParticles):
        # Note : maybe shape=(2, nParticles, 3) can also be considered ...
        super().__init__(shape=(2, 3, nParticles), dtype=float)

    @property
    def position(self):
        return self[0]

    @position.setter
    def position(self, values):
        np.copyto(self[0], values)

    @property
    def velocity(self):
        return self[1]

    @velocity.setter
    def velocity(self, values):
        np.copyto(self[1], values)


# -----------------------------------------------------------------------------
# Usage illustration
# -----------------------------------------------------------------------------

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

M = 5
nodes = np.linspace(0, 1, M)
Q = np.ones((M, M))/M
uNodes = [u0.clone(u0.values) for _ in range(M)]
out = integrate(Q, nodes, dt, uNodes)
