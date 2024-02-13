import numpy as np

from pySDC.core.Errors import DataError


class myfloat:
    """
    FD Function data type with arbitrary dimensions

    Attributes:
        values: contains the FD Function
    """

    def __init__(self, init, val=0.0):
        if isinstance(init, myfloat):
            self.values = init.values
        elif isinstance(init, float):
            self.values = init
        elif isinstance(init, int) and init == 1:
            if isinstance(val, float):
                self.values = val
            elif isinstance(val, str) and val == "random":
                self.values = np.random.random(init)[0]
            elif val is None:
                self.values = 0.0
            else:
                raise DataError(f"Type error: cannot create myfloat from val = {val}")
        else:
            raise DataError(f"Type error: cannot create myfloat from init = {init} and val = {val}")

    def copy(self, other=None):
        if other is None:  # return a copy of this vector
            return myfloat(self)
        elif isinstance(other, type(self)):  # copy the values of other into this vector
            self.values = other.values
        else:
            raise DataError("Type error: cannot copy %s to %s" % (type(other), type(self)))

    def zero(self):
        self.values = 0.0

    def __abs__(self):
        return abs(self.values)

    @property
    def n_loc_dofs(self):
        return 1

    @property
    def n_ghost_dofs(self):
        return 0

    def getSize(self):
        return 1

    def ghostUpdate(self, addv, mode):
        pass

    @property
    def numpy_array(self):
        return Exception("numpy_array not implemented")

    def isend(self, dest=None, tag=None, comm=None):
        return comm.issend(self.values, dest=dest, tag=tag)

    def irecv(self, source=None, tag=None, comm=None):
        return comm.irecv(self.values, source=source, tag=tag)

    def bcast(self, root=None, comm=None):
        comm.bcast(self.values, root=root)
        return self

    def __add__(self, other):
        if isinstance(other, type(self)):
            return myfloat(self.values + other.values)
        else:
            raise DataError("Type error: cannot add %s to %s" % (type(other), type(self)))

    def __iadd__(self, other):
        if isinstance(other, type(self)):
            self.values += other.values
            return self
        elif isinstance(other, float):
            self.values += other
            return self
        else:
            raise DataError("Type error: cannot iadd %s to %s" % (type(other), type(self)))

    def __sub__(self, other):
        if isinstance(other, type(self)):
            return myfloat(self.values - other.values)
        else:
            raise DataError("Type error: cannot sub %s from %s" % (type(other), type(self)))

    def __isub__(self, other):
        if isinstance(other, type(self)):
            self.values -= other.values
            return self
        elif isinstance(other, float):
            self.values -= other
            return self
        else:
            raise DataError("Type error: cannot isub %s to %s" % (type(other), type(self)))

    def __mul__(self, other):
        if isinstance(other, myfloat):
            return myfloat(self.values * other.values)
        elif isinstance(other, float):
            return myfloat(self.values * other)
        else:
            raise DataError("Type error: cannot rmul %s to %s" % (type(other), type(self)))

    def __rmul__(self, other):
        if isinstance(other, float):
            return myfloat(self.values * other)
        else:
            raise DataError("Type error: cannot rmul %s to %s" % (type(other), type(self)))

    def __imul__(self, other):
        if isinstance(other, myfloat):
            self.values *= other.values
            return self
        elif isinstance(other, float):
            self.values *= other
            return self
        else:
            raise DataError("Type error: cannot imul %s to %s" % (type(other), type(self)))

    def axpy(self, a, x):
        """
        Performs self.values = a*x.values+self.values
        """

        if isinstance(x, type(self)):
            self.values += a * x.values
        else:
            raise DataError("Type error: cannot add %s to %s" % (type(x), type(self)))

    def aypx(self, a, x):
        """
        Performs self.values = x.values+a*self.values
        """

        if isinstance(x, type(self)):
            self.values = a * self.values + x.values
        else:
            raise DataError("Type error: cannot add %s to %s" % (type(x), type(self)))

    def axpby(self, a, b, x):
        """
        Performs self.values = a*x.values+b*self.values
        """

        if isinstance(x, type(self)):
            self.values = a * x.values + b * self.values
        else:
            raise DataError("Type error: cannot add %s to %s" % (type(x), type(self)))

    def iadd_sub(self, other, indices):
        if indices == [0]:
            self += other

    def isub_sub(self, other, indices):
        if indices == [0]:
            self -= other

    def imul_sub(self, other, indices):
        if indices == [0]:
            self *= other

    def axpby_sub(self, a, b, x, indices):
        if indices == [0]:
            self.axpby(a, b, x)

    def axpy_sub(self, a, x, indices):
        if indices == [0]:
            self.axpy(a, x)

    def aypx_sub(self, a, x, indices):
        if indices == [0]:
            self.aypx(a, x)

    def copy_sub(self, other, indices):
        if indices == [0]:
            self.copy(other)

    def zero_sub(self, indices):
        if indices == [0]:
            self.zero()


class IMEXEXP_myfloat(object):
    def __init__(self, init=None, val=0.0):
        if isinstance(init, IMEXEXP_myfloat):
            self.expl = myfloat(init.expl)
            self.impl = myfloat(init.impl)
            self.exp = myfloat(init.exp)
            self.size = 1
        else:
            self.expl = myfloat(init, val)
            self.impl = myfloat(init, val)
            self.exp = myfloat(init, val)
            self.size = 1
