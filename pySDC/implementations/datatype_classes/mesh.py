import numpy as np
import copy as cp

from pySDC.core.Errors import DataError


class mesh(object):
    """
    Mesh data type with arbitrary dimensions

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

        # if init is another mesh, do a deepcopy (init by copy)
        if isinstance(init, mesh):
            self.values = cp.deepcopy(init.values)
        # if init is a number or a tuple of numbers, create mesh object with val as initial value
        elif isinstance(init, tuple) or isinstance(init, int):
            self.values = np.empty(init, dtype=np.float64)
            self.values[:] = val
        # something is wrong, if none of the ones above hit
        else:
            raise DataError('something went wrong during %s initialization' % type(self))

    def __add__(self, other):
        """
        Overloading the addition operator for mesh types

        Args:
            other (mesh.mesh): mesh object to be added
        Raises:
            DataError: if other is not a mesh object
        Returns:
            mesh.mesh: sum of caller and other values (self+other)
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
            other (mesh.mesh): mesh object to be subtracted
        Raises:
            DataError: if other is not a mesh object
        Returns:
            mesh.mesh: differences between caller and other values (self-other)
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
            mesh.mesh: copy of original values scaled by factor
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

    def apply_mat(self, A):
        """
        Matrix multiplication operator

        Args:
            A: a matrix

        Returns:
            mesh.mesh: component multiplied by the matrix A
        """
        if not A.shape[1] == self.values.shape[0]:
            raise DataError("ERROR: cannot apply operator %s to %s" % (A.shape[1], self))

        me = mesh(A.shape[0])
        me.values = A.dot(self.values)

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
        self.values = comm.recv(source=source, tag=tag)
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
        return comm.bcast(self, root=root)


class rhs_imex_mesh(object):
    """
    RHS data type for meshes with implicit and explicit components

    This data type can be used to have RHS with 2 components (here implicit and explicit)

    Attributes:
        impl (mesh.mesh): implicit part
        expl (mesh.mesh): explicit part
    """

    def __init__(self, init, val=0.0):
        """
        Initialization routine

        Args:
            init: can either be a tuple (one int per dimension) or a number (if only one dimension is requested)
                  or another rhs_imex_mesh object
            val (float): an initial number (default: 0.0)
        Raises:
            DataError: if init is none of the types above
        """

        # if init is another rhs_imex_mesh, do a deepcopy (init by copy)
        if isinstance(init, type(self)):
            self.impl = mesh(init.impl)
            self.expl = mesh(init.expl)
        # if init is a number or a tuple of numbers, create mesh object with None as initial value
        elif isinstance(init, tuple) or isinstance(init, int):
            self.impl = mesh(init, val=val)
            self.expl = mesh(init, val=val)
        # something is wrong, if none of the ones above hit
        else:
            raise DataError('something went wrong during %s initialization' % type(self))

    def __sub__(self, other):
        """
        Overloading the subtraction operator for rhs types

        Args:
            other (mesh.rhs_imex_mesh): rhs object to be subtracted
        Raises:
            DataError: if other is not a rhs object
        Returns:
            mesh.rhs_imex_mesh: differences between caller and other values (self-other)
        """

        if isinstance(other, rhs_imex_mesh):
            # always create new rhs_imex_mesh, since otherwise c = a - b changes a as well!
            me = rhs_imex_mesh(np.shape(self.impl.values))
            me.impl.values = self.impl.values - other.impl.values
            me.expl.values = self.expl.values - other.expl.values
            return me
        else:
            raise DataError("Type error: cannot subtract %s from %s" % (type(other), type(self)))

    def __add__(self, other):
        """
         Overloading the addition operator for rhs types

        Args:
            other (mesh.rhs_imex_mesh): rhs object to be added
        Raises:
            DataError: if other is not a rhs object
        Returns:
            mesh.rhs_imex_mesh: sum of caller and other values (self-other)
        """

        if isinstance(other, rhs_imex_mesh):
            # always create new rhs_imex_mesh, since otherwise c = a + b changes a as well!
            me = rhs_imex_mesh(np.shape(self.impl.values))
            me.impl.values = self.impl.values + other.impl.values
            me.expl.values = self.expl.values + other.expl.values
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
             mesh.rhs_imex_mesh: copy of original values scaled by factor
        """

        if isinstance(other, float):
            # always create new rhs_imex_mesh
            me = rhs_imex_mesh(np.shape(self.impl.values))
            me.impl.values = other * self.impl.values
            me.expl.values = other * self.expl.values
            return me
        else:
            raise DataError("Type error: cannot multiply %s to %s" % (type(other), type(self)))

    def apply_mat(self, A):
        """
        Matrix multiplication operator

        Args:
            A: a matrix

        Returns:
            mesh.rhs_imex_mesh: each component multiplied by the matrix A
        """

        if not A.shape[1] == self.impl.values.shape[0]:
            raise DataError("ERROR: cannot apply operator %s to %s" % (A, self.impl))
        if not A.shape[1] == self.expl.values.shape[0]:
            raise DataError("ERROR: cannot apply operator %s to %s" % (A, self.expl))

        me = rhs_imex_mesh(A.shape[1])
        me.impl.values = A.dot(self.impl.values)
        me.expl.values = A.dot(self.expl.values)

        return me


class rhs_comp2_mesh(object):
    """
    RHS data type for meshes with 2 components

    Attributes:
        comp1 (mesh.mesh): first part
        comp2 (mesh.mesh): second part
    """

    def __init__(self, init, val=0.0):
        """
        Initialization routine

        Args:
            init: can either be a tuple (one int per dimension) or a number (if only one dimension is requested)
                  or another rhs_imex_mesh object
        Raises:
            DataError: if init is none of the types above
        """

        # if init is another rhs_imex_mesh, do a deepcopy (init by copy)
        if isinstance(init, type(self)):
            self.comp1 = mesh(init.comp1)
            self.comp2 = mesh(init.comp2)
        # if init is a number or a tuple of numbers, create mesh object with None as initial value
        elif isinstance(init, tuple) or isinstance(init, int):
            self.comp1 = mesh(init, val=val)
            self.comp2 = mesh(init, val=val)
        # something is wrong, if none of the ones above hit
        else:
            raise DataError('something went wrong during %s initialization' % type(self))

    def __sub__(self, other):
        """
        Overloading the subtraction operator for rhs types

        Args:
            other (mesh.rhs_imex_mesh): rhs object to be subtracted
        Raises:
            DataError: if other is not a rhs object
        Returns:
            mesh.rhs_comp2_mesh: differences between caller and other values (self-other)
        """

        if isinstance(other, rhs_comp2_mesh):
            # always create new rhs_imex_mesh, since otherwise c = a - b changes a as well!
            me = rhs_comp2_mesh(np.shape(self.comp1.values))
            me.comp1.values = self.comp1.values - other.comp1.values
            me.comp2.values = self.comp2.values - other.comp2.values
            return me
        else:
            raise DataError("Type error: cannot subtract %s from %s" % (type(other), type(self)))

    def __add__(self, other):
        """
         Overloading the addition operator for rhs types

        Args:
            other (mesh.rhs_comp2_mesh): rhs object to be added
        Raises:
            DataError: if other is not a rhs object
        Returns:
            mesh.rhs_comp2_mesh: sum of caller and other values (self-other)
        """

        if isinstance(other, rhs_comp2_mesh):
            # always create new rhs_imex_mesh, since otherwise c = a + b changes a as well!
            me = rhs_comp2_mesh(np.shape(self.comp1.values))
            me.comp1.values = self.comp1.values + other.comp1.values
            me.comp2.values = self.comp2.values + other.comp2.values
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
             mesh.rhs_comp2_mesh: copy of original values scaled by factor
        """

        if isinstance(other, float):
            # always create new rhs_imex_mesh
            me = rhs_comp2_mesh(np.shape(self.comp1.values))
            me.comp1.values = other * self.comp1.values
            me.comp2.values = other * self.comp2.values
            return me
        else:
            raise DataError("Type error: cannot multiply %s to %s" % (type(other), type(self)))

    def apply_mat(self, A):
        """
        Matrix multiplication operator

        Args:
            A: a matrix

        Returns:
            mesh.rhs_comp2_mesh: each component multiplied by the matrix A
        """

        if not A.shape[1] == self.comp1.values.shape[0]:
            raise DataError("ERROR: cannot apply operator %s to %s" % (A, self.comp1))
        if not A.shape[1] == self.comp2.values.shape[0]:
            raise DataError("ERROR: cannot apply operator %s to %s" % (A, self.comp2))

        me = rhs_comp2_mesh(A.shape[1])
        me.comp1.values = A.dot(self.comp1.values)
        me.comp2.values = A.dot(self.comp2.values)

        return me
