import numpy as np

from pySDC.core.Errors import DataError


class DCT_Vector:
    """
    DCT Function data type

    Arguments::
    ----------
        init: the size of the DCT_Vector, or another DCT_Vector to copy from or a numpy array to copy from
        val (optional): A value to initialize the DCT_Vector. Can be a float or the string 'random' to initialize it as a random vector. Defaults to 0.0.


    Attributes:
    ----------
        values: contains the DCT Function as a numpy array

    """

    def __init__(self, init, val=0.0):
        if isinstance(init, DCT_Vector):
            self.values = init.values.copy()
        elif isinstance(init, np.ndarray):
            self.values = init.copy()
        elif isinstance(init, int):
            self.values = np.ndarray(init)
            if isinstance(val, float):
                self.values.fill(val)
            elif isinstance(val, str) and val == "random":
                self.values = np.random.random(init)
            elif val is None:
                pass  # leave values uninitialized
            else:
                raise DataError(f"Type error: cannot create DCT_Vector from val = {val}")
        else:
            raise DataError(f"Type error: cannot create DCT_Vector from init = {init}")

    def copy(self, other=None):
        """
        Returns a copy of this vector or copies the values of another vector into this vector.
        If other is None, returns a copy of this vector. If other is a DCT_Vector, copies the values of other into this vector.
        """
        if other is None:  # return a copy of this vector
            return DCT_Vector(self)
        elif isinstance(other, type(self)):  # copy the values of other into this vector
            self.values[:] = other.values[:]
        else:
            raise DataError("Type error: cannot copy %s to %s" % (type(other), type(self)))

    def zero(self):
        """
        Sets all values to zero.
        """
        self.values *= 0.0

    def __abs__(self):
        """
        Returns the l2 norm of the DCT_Vector divided by the square root of the size of the DCT_Vector.
        This normalization is used to make the norm of a DCT_Vector of all ones equal to 1.
        It is useful when solving PDES, since it makes the norm of the solution independent from the mesh size.
        """
        return np.linalg.norm(self.values) / np.sqrt(self.values.size)

    def getSize(self):
        return self.values.size

    def ghostUpdate(self, addv, mode):
        pass

    @property
    def numpy_array(self):
        return self.values

    def isend(self, dest=None, tag=None, comm=None):
        return comm.Issend(self.values[:], dest=dest, tag=tag)

    def irecv(self, source=None, tag=None, comm=None):
        return comm.Irecv(self.values[:], source=source, tag=tag)

    def bcast(self, root=None, comm=None):
        comm.Bcast(self.values[:], root=root)
        return self

    def __sub__(self, other):
        if isinstance(other, type(self)):
            me = DCT_Vector(self)
            me -= other
            return me
        else:
            raise DataError("Type error: cannot sub %s from %s" % (type(other), type(self)))

    def __iadd__(self, other):
        if isinstance(other, type(self)):
            self.values += other.values
            return self
        elif isinstance(other, float):
            self.values += other
            return self
        else:
            raise DataError("Type error: cannot iadd %s to %s" % (type(other), type(self)))

    def __isub__(self, other):
        if isinstance(other, type(self)):
            self.values -= other.values
            return self
        else:
            raise DataError("Type error: cannot isub %s to %s" % (type(other), type(self)))

    def __imul__(self, other):
        if isinstance(other, float) or isinstance(other, np.ndarray):
            self.values *= other
            return self
        elif isinstance(other, DCT_Vector):
            self.values *= other.values
            return self
        else:
            raise DataError("Type error: cannot imul %s to %s" % (type(other), type(self)))

    def axpy(self, a, x):
        """
        Performs self.values = a * x.values + self.values
        """

        if isinstance(x, type(self)):
            self.values += a * x.values
        else:
            raise DataError("Type error: cannot add %s to %s" % (type(x), type(self)))

    def aypx(self, a, x):
        """
        Performs self.values = x.values + a * self.values
        """

        if isinstance(x, type(self)):
            self.values *= a
            self.values += x.values
        else:
            raise DataError("Type error: cannot add %s to %s" % (type(x), type(self)))
