from mpi4py import MPI

import numpy as np
from pmesh.pm import ParticleMesh

from pySDC.core.Errors import DataError


class pmesh_datatype(object):
    """
    Mesh data type with arbitrary dimensions

    This data type can be used whenever structured data with a single unknown per point in space is required

    Attributes:
        values (np.ndarray): contains the ndarray of the values
    """

    def __init__(self, init=None, val=0.0):
        """
        Initialization routine

        Args:
            init: can either be a tuple (one int per dimension) or a number (if only one dimension is requested)
                  or another mesh object
        Raises:
            DataError: if init is none of the types above
        """

        # if init is another mesh, do a deepcopy (init by copy)
        if isinstance(init, ParticleMesh):
            self.values = init.create(type='real', value=val)
        elif isinstance(init, type(self)):
            # self.values = init.values.pm.create(type='real', value=init.values)
            self.values = init.values
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

        if isinstance(other, type(self)):
            # always create new mesh, since otherwise c = a + b changes a as well!
            me = pmesh_datatype(self)
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

        if isinstance(other, type(self)):
            # always create new mesh, since otherwise c = a - b changes a as well!
            me = pmesh_datatype(self)
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
            me = pmesh_datatype(self)
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

        comm = self.values.pm.comm
        if comm is not None:
            if comm.Get_size() > 1:
                global_absval = comm.allreduce(sendobj=local_absval, op=MPI.MAX)
            else:
                global_absval = local_absval
        else:
            global_absval = local_absval

        return global_absval

        # # take absolute values of the mesh values
        # absval = abs(self.values)
        # # return maximum
        # return np.amax(absval)

    # def apply_mat(self, A):
    #     """
    #     Matrix multiplication operator
    #
    #     Args:
    #         A: a matrix
    #
    #     Returns:
    #         mesh.mesh: component multiplied by the matrix A
    #     """
    #     if not A.shape[1] == self.values.shape[0]:
    #         raise DataError("ERROR: cannot apply operator %s to %s" % (A.shape[1], self))
    #
    #     me = mesh(A.shape[0])
    #     me.values = A.dot(self.values)
    #
    #     return me

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

        comm.send(self.values.value, dest=dest, tag=tag)
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
        return comm.isend(self.values.value, dest=dest, tag=tag)

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
        self.values = self.values.pm.create(type='real', value=comm.recv(source=source, tag=tag))
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
        me = pmesh_datatype(self)
        me.values = self.values.pm.create(type='real', value=comm.bcast(self.values.value, root=root))
        return me


class rhs_imex_pmesh(object):
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
            self.impl = pmesh_datatype(init.impl)
            self.expl = pmesh_datatype(init.expl)
        elif isinstance(init, ParticleMesh):
            self.impl = pmesh_datatype(init)
            self.expl = pmesh_datatype(init)
        # # if init is a number or a tuple of numbers, create mesh object with None as initial value
        # elif isinstance(init, tuple) or isinstance(init, int):
        #     self.impl = pmesh_datatype(init, val=val)
        #     self.expl = mesh(init, val=val)
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

        if isinstance(other, type(self)):
            # always create new rhs_imex_mesh, since otherwise c = a - b changes a as well!
            me = rhs_imex_pmesh(self)
            me.impl = self.impl - other.impl
            me.expl = self.expl - other.expl
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

        if isinstance(other, type(self)):
            # always create new rhs_imex_mesh, since otherwise c = a + b changes a as well!
            me = rhs_imex_pmesh(self)
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
             mesh.rhs_imex_mesh: copy of original values scaled by factor
        """

        if isinstance(other, float):
            # always create new rhs_imex_mesh
            me = rhs_imex_pmesh(self)
            me.impl = other * self.impl
            me.expl = other * self.expl
            return me
        else:
            raise DataError("Type error: cannot multiply %s to %s" % (type(other), type(self)))
    #
    # def apply_mat(self, A):
    #     """
    #     Matrix multiplication operator
    #
    #     Args:
    #         A: a matrix
    #
    #     Returns:
    #         mesh.rhs_imex_mesh: each component multiplied by the matrix A
    #     """
    #
    #     if not A.shape[1] == self.impl.values.shape[0]:
    #         raise DataError("ERROR: cannot apply operator %s to %s" % (A, self.impl))
    #     if not A.shape[1] == self.expl.values.shape[0]:
    #         raise DataError("ERROR: cannot apply operator %s to %s" % (A, self.expl))
    #
    #     me = rhs_imex_mesh(A.shape[1])
    #     me.impl.values = A.dot(self.impl.values)
    #     me.expl.values = A.dot(self.expl.values)
    #
    #     return me

