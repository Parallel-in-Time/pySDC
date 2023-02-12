#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 17:59:56 2023

@author: cpf5546
"""
from pySDC.playgrounds.datatypes.base import DataType

import numpy as np
from petsc4py import PETSc
try:
    from mpi4py import MPI
except ImportError:
    MPI = None

# -----------------------------------------------------------------------------
# Specific implementations (in pySDC/implementations)
# -----------------------------------------------------------------------------
class Mesh(DataType, np.ndarray):
    # Note : priority to methods from DataType class

    # Space communicator
    _comm = None if MPI is None else MPI.COMM_WORLD

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
        self.setArray(values)

    def clone(self, values=None, copy=True):
        # TODO : implementation
        if values is None:
            return PETScMesh()
        if not copy:
            pass
        return self.copy()

    # Additional redefinitions
    def __abs__(self):
        return self.norm(3)


class Particles(Mesh):
    # TODO : understand what p and q are in the original implementation

    def __new__(cls, nParticles):
        # Note : maybe shape=(2, nParticles, 3) can also be considered ...
        return super().__new__(cls, shape=(2, 3, nParticles), dtype=float)

    @property
    def nParticles(self):
        return self.shape[2]

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
