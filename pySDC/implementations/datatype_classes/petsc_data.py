from pySDC.core.Errors import DataError

from petsc4py import PETSc


class petsc_data(object):
    """
    Wrapper for PETSc Vectors

    Attributes:
        values (np.ndarray): contains the ndarray of the values
    """

    def __init__(self, init=None, val=None):
        """
        Initialization routine

        Args:
            init: can either be a tuple (one int per dimension) or a number (if only one dimension is requested)
                  or another mesh object
            val: initial value (default: None)
        Raises:
            DataError: if init is none of the types above
        """

        # if init is another petsc data type, do a copy (init by copy)
        if isinstance(init, type(self)):
            self.values = init.values.copy()
        # if init is a DMDA, create an empty object
        elif isinstance(init, PETSc.DMDA):
            self.values = init.createGlobalVec()
        # something is wrong, if none of the ones above hit
        else:
            raise DataError('something went wrong during %s initialization' % type(self))

    def __add__(self, other):
        """
        Overloading the addition operator

        Args:
            other: PETSc object to be added
        Raises:
            DataError: if other is not a mesh object
        Returns:
            sum of caller and other values (self+other)
        """

        if isinstance(other, type(self)):
            me = petsc_data(self)
            me.values += other.values
            return me
        else:
            raise DataError("Type error: cannot add %s to %s" % (type(other), type(self)))

    def __sub__(self, other):
        """
        Overloading the subtraction operator

        Args:
            other: PETSc object to be subtracted
        Raises:
            DataError: if other is not a mesh object
        Returns:
            differences between caller and other values (self-other)
        """

        if isinstance(other, type(self)):
            me = petsc_data(self)
            me.values -= other.values
            return me
        else:
            raise DataError("Type error: cannot subtract %s from %s" % (type(other), type(self)))

    def __rmul__(self, other):
        """
        Overloading the right multiply by factor operator

        Args:
            other (float): factor
        Raises:
            DataError: is other is not a float
        Returns:
            copy of original values scaled by factor
        """

        if isinstance(other, float) or isinstance(other, complex):
            me = petsc_data(self)
            me.values *= other
            return me
        else:
            raise DataError("Type error: cannot multiply %s to %s" % (type(other), type(self)))

    def __abs__(self):
        """
        Overloading the abs operator

        Returns:
            float: absolute maximum
        """
        # return maximum
        return self.values.norm(PETSc.NormType.NORM_INFINITY)

    def apply_mat(self, A):
        """
        Matrix multiplication operator

        Args:
            A: a matrix

        Returns:
            component multiplied by the matrix A
        """

        me = petsc_data(self)
        A.mult(self.values, me.values)
        return me

    def send(self, dest=None, tag=None, comm=None):
        """
        Routine for sending data forward in time (blocking)

        Args:
            dest (int): target rank
            tag (int): communication tag
            comm: communicator

        Returns:
            None
        """

        comm.send(self.values.getArray(), dest=dest, tag=tag)
        return None

    def isend(self, dest=None, tag=None, comm=None):
        """
        Routine for sending data forward in time (non-blocking)

        Args:
            dest (int): target rank
            tag (int): communication tag
            comm: communicator

        Returns:
            request handle
        """
        return comm.isend(self.values.getArray(), dest=dest, tag=tag)

    def recv(self, source=None, tag=None, comm=None):
        """
        Routine for receiving in time

        Args:
            source (int): source rank
            tag (int): communication tag
            comm: communicator

        Returns:
            None
        """
        self.values.setArray(comm.recv(source=source, tag=tag))
        return None

    def bcast(self, root=None, comm=None):
        """
        Routine for broadcasting values

        Args:
            root (int): process with value to broadcast
            comm: communicator

        Returns:
            broadcasted values
        """

        me = petsc_data(self)
        me.values.setArray(comm.bcast(self.values.getArray(), root=root))
        return me


class rhs_imex_petsc_data(object):
    """
    Wrapper for PETSc Vectors with two components

    Attributes:
        values (np.ndarray): contains the ndarray of the values
    """

    def __init__(self, init=None, val=None):
        """
        Initialization routine

        Args:
            init: can either be a tuple (one int per dimension) or a number (if only one dimension is requested)
                  or another mesh object
            val: initial value (default: None)
        Raises:
            DataError: if init is none of the types above
        """

        # if init is another petsc data type, do a copy (init by copy)
        if isinstance(init, type(self)):
            self.impl = petsc_data(init.impl)
            self.expl = petsc_data(init.expl)
        # if init is a DMDA, create an empty object
        elif isinstance(init, PETSc.DMDA):
            self.impl = petsc_data(init)
            self.expl = petsc_data(init)
        # something is wrong, if none of the ones above hit
        else:
            raise DataError('something went wrong during %s initialization' % type(self))

    def __add__(self, other):
        """
        Overloading the addition operator

        Args:
            other: PETSc object to be added
        Raises:
            DataError: if other is an unexpected object
        Returns:
            sum of caller and other values (self+other)
        """

        if isinstance(other, type(self)):
            me = rhs_imex_petsc_data(self)
            me.impl.values += other.impl.values
            me.expl.values += other.expl.values
            return me
        else:
            raise DataError("Type error: cannot add %s to %s" % (type(other), type(self)))

    def __sub__(self, other):
        """
        Overloading the subtraction operator

        Args:
            other: PETSc object to be subtracted
        Raises:
            DataError: if other is an unexpected object
        Returns:
            differences between caller and other values (self-other)
        """

        if isinstance(other, type(self)):
            me = rhs_imex_petsc_data(self)
            me.impl.values -= other.impl.values
            me.expl.values -= other.expl.values
            return me
        else:
            raise DataError("Type error: cannot subtract %s from %s" % (type(other), type(self)))

    def __rmul__(self, other):
        """
        Overloading the right multiply by factor operator

        Args:
            other (float): factor
        Raises:
            DataError: is other is not a float
        Returns:
            copy of original values scaled by factor
        """

        if isinstance(other, float) or isinstance(other, complex):
            me = rhs_imex_petsc_data(self)
            me.impl.values *= other
            me.expl.values *= other
            return me
        else:
            raise DataError("Type error: cannot multiply %s to %s" % (type(other), type(self)))

    def apply_mat(self, A):
        """
        Matrix multiplication operator

        Args:
            A: a matrix

        Returns:
            component multiplied by the matrix A
        """

        me = rhs_imex_petsc_data(self)
        A.mult(self.impl.values, me.impl.values)
        A.mult(self.expl.values, me.expl.values)
        return me