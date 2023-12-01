from dolfinx import fem
import numpy as np
from petsc4py import PETSc

from pySDC.core.Errors import DataError


class FEniCSx_Vector(object):
    """
    FEniCSx Function data type with arbitrary dimensions

    Attributes:
        values: contains the fenicsx Function
    """

    def __init__(self, init=None, val=0.0):
        """
        Initialization routine

        Attribute:
            values: a dolfinx.fem.Function

        Args:
            init: can either be another FEniCSx_Vector object to be copied, a fem.Function to be copied into values
                  or a fem.FunctionSpace (with a constant value val to be assigned to the fem.Function)
            val: initial value (default: 0.0)
        Raises:
            DataError: if init is none of the types above
        """

        if isinstance(init, type(self)):
            self.values = init.values.copy()
        elif isinstance(init, fem.Function):
            self.values = init.copy()
        elif isinstance(init, fem.FunctionSpace):
            self.values = fem.Function(init)
            if isinstance(val, str) and val == "random":
                self.values.x.array[:] = np.random.random(self.values.x.array.shape)[:]
            elif isinstance(val, float):
                self.values.vector.set(val)
            elif isinstance(val, PETSc.Vec):
                self.values.vector.setArray(val)
            else:
                raise DataError("something went wrong during %s initialization" % type(init))
        else:
            raise DataError("something went wrong during %s initialization" % type(init))

    def copy(self, other=None):
        if other is None:  # return a copy of this vector
            return FEniCSx_Vector(self)
        elif isinstance(other, type(self)):  # copy the values of other into this vector
            self.values.vector.setArray(other.values.vector)
        else:
            raise DataError("Type error: cannot copy %s to %s" % (type(other), type(self)))

    def zero(self):
        """
        Set to zero.
        """
        with self.values.vector.localForm() as loc_self:
            loc_self.set(0.0)

    def __abs__(self):
        self.values.vector.normBegin()
        norm_L2 = self.values.vector.normEnd()
        norm_L2 /= np.sqrt(self.getSize())

        return norm_L2

    def dot(self, other):
        self.values.vector.dotBegin(other.values.vector)
        res = self.values.vector.dotEnd(other.values.vector)
        return res

    def interpolate(self, other):
        if isinstance(other, FEniCSx_Vector):
            self.values.interpolate(
                other.values,
                nmm_interpolation_data=fem.create_nonmatching_meshes_interpolation_data(
                    self.values.function_space.mesh._cpp_object, self.values.function_space.element, other.values.function_space.mesh._cpp_object
                ),
            )
        else:
            raise DataError("Type error: cannot interpolate %s to %s" % (type(other), type(self)))

    def restrict(self, other):
        self.interpolate(other)  # this is bad, should use restriction operator instead

    def prolong(self, other):
        self.interpolate(other)

    @property
    def n_loc_dofs(self):
        return self.values.vector.getSizes()[0]

    @property
    def n_ghost_dofs(self):
        return self.values.x.array.size - self.n_loc_dofs

    def getSize(self):
        return self.values.vector.getSize()

    def ghostUpdate(self, addv, mode):
        self.values.vector.ghostUpdate(addv, mode)

    def get_numpy_array(self):
        return self.values.x.array

    def isend(self, dest=None, tag=None, comm=None):
        return comm.Issend(self.values.vector.getArray(), dest=dest, tag=tag)

    def irecv(self, source=None, tag=None, comm=None):
        return comm.Irecv(self.values.vector.getArray(), source=source, tag=tag)

    def bcast(self, root=None, comm=None):
        comm.Bcast(self.values.vector.getArray(), root=root)
        return self

    def __add__(self, other):
        if isinstance(other, type(self)):
            me = FEniCSx_Vector(self)
            me.values.vector.axpy(1.0, other.values.vector)
            return me
        else:
            raise DataError("Type error: cannot add %s to %s" % (type(other), type(self)))

    def __iadd__(self, other):
        if isinstance(other, type(self)):
            self.values.vector.axpy(1.0, other.values.vector)
            return self
        else:
            raise DataError("Type error: cannot iadd %s to %s" % (type(other), type(self)))

    def __sub__(self, other):
        if isinstance(other, type(self)):
            me = FEniCSx_Vector(self)
            me.values.vector.axpy(-1.0, other.values.vector)
            return me
        else:
            raise DataError("Type error: cannot sub %s from %s" % (type(other), type(self)))

    def __isub__(self, other):
        if isinstance(other, type(self)):
            self.values.vector.axpy(-1.0, other.values.vector)
            return self
        else:
            raise DataError("Type error: cannot isub %s to %s" % (type(other), type(self)))

    def __mul__(self, other):
        if isinstance(other, FEniCSx_Vector) or isinstance(other, float):
            me = FEniCSx_Vector(self)
            me *= other
            return me

        else:
            raise DataError("Type error: cannot rmul %s to %s" % (type(other), type(self)))

    def __rmul__(self, other):
        if isinstance(other, float):
            me = FEniCSx_Vector(self)
            me *= other
            return me
        else:
            raise DataError("Type error: cannot rmul %s to %s" % (type(other), type(self)))

    def __imul__(self, other):
        if isinstance(other, float):
            self.values.vector.scale(other)
            return self
        elif isinstance(other, FEniCSx_Vector):
            self.values.x.array[:] *= other.values.x.array[:]
        else:
            raise DataError("Type error: cannot imul %s to %s" % (type(other), type(self)))

    def axpy(self, a, x):
        """
        Performs self.values = a*x.values+self.values
        """

        if isinstance(x, type(self)):
            self.values.vector.axpy(a, x.values.vector)
        else:
            raise DataError("Type error: cannot add %s to %s" % (type(x), type(self)))

    def aypx(self, a, x):
        """
        Performs self.values = x.values+a*self.values
        """

        if isinstance(x, type(self)):
            self.values.vector.aypx(a, x.values.vector)
        else:
            raise DataError("Type error: cannot add %s to %s" % (type(x), type(self)))

    def axpby(self, a, b, x):
        """
        Performs self.values = a*x.values+b*self.values
        """

        if isinstance(x, type(self)):
            self.values.vector.axpby(a, b, x.values.vector)
        else:
            raise DataError("Type error: cannot add %s to %s" % (type(x), type(self)))
