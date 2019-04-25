from mpi4py import MPI
import numpy as np

from pySDC.core.Errors import DataError
from mpi4py_fft import newDistArray, PFFT


class fft_datatype(object):
    """
    Mesh data type with arbitrary dimensions, will contain PMESH values and communicator

    Attributes:
        values (np.ndarray): contains the ndarray of the values
        comm: MPI communicator or None
    """

    def __init__(self, init=None, val=0.0):
        """
        Initialization routine

        Args:
            init: another pmesh_datatype or a tuple containing the communicator and the local dimensions
            val (float): an initial number (default: 0.0)
        Raises:
            DataError: if init is none of the types above
        """
        if isinstance(init, fft_datatype):
            self.fft = init.fft
            self.values = init.values.copy()
        elif isinstance(init, tuple) and isinstance(init[0], PFFT):
            self.fft = init[0]
            self.values = newDistArray(self.fft, init[1], val=val)
        # something is wrong, if none of the ones above hit
        else:
            raise DataError('something went wrong during %s initialization' % type(self))

    def __add__(self, other):
        """
        Overloading the addition operator for mesh types

        Args:
            other: mesh object to be added
        Raises:
            DataError: if other is not a mesh object
        Returns:
            sum of caller and other values (self+other)
        """

        if isinstance(other, type(self)):
            # always create new mesh, since otherwise c = a + b changes a as well!
            me = fft_datatype(self)
            me.values = self.values + other.values
            return me
        else:
            raise DataError("Type error: cannot add %s to %s" % (type(other), type(self)))

    def __sub__(self, other):
        """
        Overloading the subtraction operator for mesh types

        Args:
            other: mesh object to be subtracted
        Raises:
            DataError: if other is not a mesh object
        Returns:
            differences between caller and other values (self-other)
        """

        if isinstance(other, type(self)):
            # always create new mesh, since otherwise c = a - b changes a as well!
            me = fft_datatype(self)
            me.values = self.values - other.values
            return me
        else:
            raise DataError("Type error: cannot subtract %s from %s" % (type(other), type(self)))

    def __rmul__(self, other):
        """
        Overloading the right multiply by factor operator for mesh types

        Args:
            other (float): factor
        Raises:
            DataError: is other is not a float
        Returns:
            copy of original values scaled by factor
        """

        if isinstance(other, float) or isinstance(other, complex):
            # always create new mesh, since otherwise c = f*a changes a as well!
            me = fft_datatype(self)
            me.values = other * self.values
            return me
        else:
            raise DataError("Type error: cannot multiply %s to %s" % (type(other), type(self)))

    def __abs__(self):
        """
        Overloading the abs operator for mesh types

        Returns:
            float: absolute maximum of all mesh values
        """
        # take absolute values of the mesh values
        local_absval = np.amax(abs(self.values))

        comm = self.fft.subcomm
        if comm is not None:
            if np.any([c.Get_size() > 1 for c in comm]):
                global_absval = 0.0
                for c in comm:
                    global_absval = max(c.allreduce(sendobj=local_absval, op=MPI.MAX), global_absval)  # TODO: is this correct?
            else:
                global_absval = local_absval
        else:
            global_absval = local_absval

        return global_absval

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

        comm.send(self.values, dest=dest, tag=tag)
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
        return comm.isend(self.values, dest=dest, tag=tag)

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
        self.values[:] = comm.recv(source=source, tag=tag)
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
        me = fft_datatype(self)
        me.values[:] = comm.bcast(self.values, root=root)
        return me


class rhs_imex_fft(object):
    """
    RHS data type for PMESH datatypes with implicit and explicit components

    This data type can be used to have RHS with 2 components (here implicit and explicit)

    Attributes:
        impl: implicit part as pmesh_datatype
        expl: explicit part as pmesh_datatype
    """

    def __init__(self, init, val=0.0):
        """
        Initialization routine

        Args:
            init: another pmesh_datatype or a tuple containing the communicator and the local dimensions
            val (float): an initial number (default: 0.0)
        Raises:
            DataError: if init is none of the types above
        """
        if isinstance(init, type(self)):
            self.impl = fft_datatype(init.impl)
            self.expl = fft_datatype(init.expl)
        elif isinstance(init, tuple) and isinstance(init[0], PFFT):
            self.impl = fft_datatype(init, val=val)
            self.expl = fft_datatype(init, val=val)
        # something is wrong, if none of the ones above hit
        else:
            raise DataError('something went wrong during %s initialization' % type(self))

    def __sub__(self, other):
        """
        Overloading the subtraction operator for rhs types

        Args:
            other: rhs object to be subtracted
        Raises:
            DataError: if other is not a rhs object
        Returns:
            differences between caller and other values (self-other)
        """

        if isinstance(other, type(self)):
            # always create new rhs_imex_mesh, since otherwise c = a - b changes a as well!
            me = rhs_imex_fft(self)
            me.impl = self.impl - other.impl
            me.expl = self.expl - other.expl
            return me
        else:
            raise DataError("Type error: cannot subtract %s from %s" % (type(other), type(self)))

    def __add__(self, other):
        """
         Overloading the addition operator for rhs types

        Args:
            other: rhs object to be added
        Raises:
            DataError: if other is not a rhs object
        Returns:
            sum of caller and other values (self-other)
        """

        if isinstance(other, type(self)):
            # always create new rhs_imex_mesh, since otherwise c = a + b changes a as well!
            me = rhs_imex_fft(self)
            me.impl = self.impl + other.impl
            me.expl = self.expl + other.expl
            return me
        else:
            raise DataError("Type error: cannot add %s to %s" % (type(other), type(self)))

    def __rmul__(self, other):
        """
        Overloading the right multiply by factor operator for mesh types

        Args:
            other (float): factor
        Raises:
            DataError: is other is not a float
        Returns:
             copy of original values scaled by factor
        """

        if isinstance(other, float):
            # always create new rhs_imex_mesh
            me = rhs_imex_fft(self)
            me.impl = other * self.impl
            me.expl = other * self.expl
            return me
        else:
            raise DataError("Type error: cannot multiply %s to %s" % (type(other), type(self)))
