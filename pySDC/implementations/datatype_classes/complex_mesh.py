import numpy as np

from pySDC.core.Errors import DataError


class mesh(object):
    """
    Mesh data type with arbitrary dimensions and complex datay

    This data type can be used whenever structured data with a single unknown per point in space is required

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

        # if init is another mesh, do a copy (init by copy)
        if isinstance(init, mesh):
            self.values = np.copy(init.values)
        # if init is a number or a tuple of numbers, create mesh object with val as initial value
        elif isinstance(init, tuple) or isinstance(init, int):
            self.values = np.empty(init, dtype=np.complex)
            self.values[:] = val
        # something is wrong, if none of the ones above hit
        else:
            raise DataError('something went wrong during %s initialization' % type(self))

    def __add__(self, other):
        """
        Overloading the addition operator for mesh types

        Args:
            other (complex_mesh.mesh): mesh object to be added
        Raises:
            DataError: if other is not a mesh object
        Returns:
            complex_mesh.mesh: sum of caller and other values (self+other)
        """

        if isinstance(other, mesh):
            # always create new mesh, since otherwise c = a + b changes a as well!
            me = mesh(np.shape(self.values))
            me.values = self.values + other.values
            return me
        else:
            raise DataError("Type error: cannot add %s to %s" % (type(other), type(self)))

    def __sub__(self, other):
        """
        Overloading the subtraction operator for mesh types

        Args:
            other (complex_mesh.mesh): mesh object to be subtracted
        Raises:
            DataError: if other is not a mesh object
        Returns:
            complex_mesh.mesh: differences between caller and other values (self-other)
        """

        if isinstance(other, mesh):
            # always create new mesh, since otherwise c = a - b changes a as well!
            me = mesh(np.shape(self.values))
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
            complex_mesh.mesh: copy of original values scaled by factor
        """

        if isinstance(other, float) or isinstance(other, complex):
            # always create new mesh, since otherwise c = f*a changes a as well!
            me = mesh(np.shape(self.values))
            me.values = self.values * other
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
        absval = abs(self.values)
        # return maximum
        return np.amax(absval)

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
        return comm.Issend(self.values, dest=dest, tag=tag)

    def irecv(self, source=None, tag=None, comm=None):
        """
        Routine for receiving in time

        Args:
            source (int): source rank
            tag (int): communication tag
            comm: communicator

        Returns:
            None
        """
        return comm.Irecv(self.values, source=source, tag=tag)


class rhs_imex_mesh(object):
    """
    RHS data type for meshes with implicit and explicit components

    This data type can be used to have RHS with 2 components (here implicit and explicit)

    Attributes:
        impl (complex_mesh.mesh): implicit part
        expl (complex_mesh.mesh): explicit part
    """

    def __init__(self, init):
        """
        Initialization routine

        Args:
            init: can either be a tuple (one int per dimension) or a number (if only one dimension is requested)
                  or another rhs_imex_mesh object
        Raises:
            DataError: if init is none of the types above
        """

        # if init is another rhs_imex_mesh, do a copy (init by copy)
        if isinstance(init, type(self)):
            self.impl = mesh(init.impl)
            self.expl = mesh(init.expl)
        # if init is a number or a tuple of numbers, create mesh object with None as initial value
        elif isinstance(init, tuple) or isinstance(init, int):
            self.impl = mesh(init)
            self.expl = mesh(init)
        # something is wrong, if none of the ones above hit
        else:
            raise DataError('something went wrong during %s initialization' % type(self))
