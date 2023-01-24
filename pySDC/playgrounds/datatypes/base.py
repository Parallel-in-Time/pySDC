#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 17:58:37 2023

@author: cpf5546
"""
# -----------------------------------------------------------------------------
# Base core implementation
# -----------------------------------------------------------------------------
try:
    from mpi4py import MPI
except ImportError:
    MPI = None


class DataType(object):
    """Base datatype class for solution"""

    # Methods used for time communications
    def isend(self, dest=None, tag=None, comm=None):
        return comm.Issend(self.values, dest=dest, tag=tag)

    def irecv(self, source=None, tag=None, comm=None):
        return comm.Irecv(self.values, source=source, tag=tag)

    def bcast(self, root=None, comm=None):
        comm.Bcast(self.values, root=root)
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
            obj.parts = tuple(obj.__getattribute__(name) for name in obj._parts)
            return obj

    @property
    def partNames(self):
        return self._parts

    def __setattr__(self, name, value):
        if name in self._parts:
            raise ValueError(f'{name} is read-only')
        super().__setattr__(name, value)

    def __repr__(self):
        return '{' + ',\n '.join(
            f'{c}: {getattr(self, c).__repr__()}' for c in self._parts) + '}'

    def toDataType(self):
        parts = self.parts
        out = self._dataType.clone(values=parts[0].values)
        for c in parts[1:]:
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
