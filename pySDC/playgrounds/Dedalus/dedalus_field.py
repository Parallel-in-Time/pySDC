import numpy as np
from mpi4py import MPI

from dedalus import public as de

from pySDC.core.Errors import DataError


class dedalus_field(object):
    """
    Dedalus data type

    This data type can be used whenever structured data with a single unknown per point in space is required

    Attributes:
        values: contains the domain field
        domain: contains the domain
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
        if isinstance(init, de.Domain):
            self.values = [init.new_field()]
        elif isinstance(init, type(self)):
            self.values = []
            for f in init.values:
                self.values.append(f.domain.new_field())
                self.values[-1]['g'] = f['g']
        elif isinstance(init, tuple):
            self.values = []
            for i in range(init[1]):
                self.values.append(init[0].new_field())

        # elif isinstance(init, type(self)):
        #     if hasattr(init, 'values'):
        #         self.values = init.values.domain.new_field()
        #         self.values['g'] = init.values['g']
        #     elif hasattr(init, 'list_of_values'):
        #         self.list_of_values = []
        #         for f in init.list_of_values:
        #             self.list_of_values.append(f.domain.new_field())
        #             self.list_of_values[-1]['g'] = f['g']
        #     else:
        #         raise DataError('something went really wrong during %s initialization' % type(self))
        # elif isinstance(init, tuple):
        #     self.list_of_values = []
        #     for i in range(init[0]):
        #         self.list_of_values.append(init[1].new_field())
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
            me = dedalus_field(other)
            for l in range(len(me.values)):
                me.values[l]['g'] = self.values[l]['g'] + other.values[l]['g']
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
            me = dedalus_field(other)
            for l in range(len(me.values)):
                me.values[l]['g'] = self.values[l]['g'] - other.values[l]['g']
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

        if isinstance(other, float):
            # always create new mesh, since otherwise c = a * factor changes a as well!
            me = dedalus_field(self)
            for l in range(len(me.values)):
                me.values[l]['g'] = other * self.values[l]['g']
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
        local_absval = np.amax([abs(f['g']) for f in self.values])

        comm = self.values[0].domain.distributor.comm
        if comm is not None:
            if comm.Get_size() > 1:
                global_absval = comm.allreduce(sendobj=local_absval, op=MPI.MAX)
            else:
                global_absval = local_absval
        else:
            global_absval = local_absval

        return global_absval

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
    #     me = dedalus_field(self.values.domain)
    #     me.values['g'] = A.dot(self.values['g'])
    #
    #     return me

    # def send(self, dest=None, tag=None, comm=None):
    #     """
    #     Routine for sending data forward in time (blocking)
    #
    #     Args:
    #         dest (int): target rank
    #         tag (int): communication tag
    #         comm: communicator
    #
    #     Returns:
    #         None
    #     """
    #
    #     comm.send(self.values['g'], dest=dest, tag=tag)
    #     return None

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
        req = None
        for data in self.values:
            if req is not None:
                req.Free()
            req = comm.Issend(data['g'][:], dest=dest, tag=tag)
        return req

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
        req = None
        for data in self.values:
            if req is not None:
                req.Free()
            req = comm.Irecv(data['g'], source=source, tag=tag)
        return req
        # return comm.Irecv(self.values['g'][:], source=source, tag=tag)

    def bcast(self, root=None, comm=None):
        """
        Routine for broadcasting values

        Args:
            root (int): process with value to broadcast
            comm: communicator

        Returns:
            broadcasted values
        """
        me = dedalus_field(self)
        for l in range(len(me.values)):
            me.values[l]['g'] = comm.bcast(self.values[l]['g'], root=root)
        return me


class rhs_imex_dedalus_field(object):
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
                  or another rhs_imex_field object
            val (float): an initial number (default: 0.0)
        Raises:
            DataError: if init is none of the types above
        """

        # if init is another rhs_imex_field, do a deepcopy (init by copy)
        if isinstance(init, type(self)):
            self.impl = dedalus_field(init.impl)
            self.expl = dedalus_field(init.expl)
        elif isinstance(init, de.Domain) or isinstance(init, tuple):
            self.impl = dedalus_field(init)
            self.expl = dedalus_field(init)
        else:
            raise DataError('something went wrong during %s initialization' % type(self))

    def __sub__(self, other):
        """
        Overloading the subtraction operator for rhs types

        Args:
            other (mesh.rhs_imex_field): rhs object to be subtracted
        Raises:
            DataError: if other is not a rhs object
        Returns:
            mesh.rhs_imex_field: differences between caller and other values (self-other)
        """

        if isinstance(other, rhs_imex_dedalus_field):
            # always create new rhs_imex_field, since otherwise c = a - b changes a as well!
            me = rhs_imex_dedalus_field(self.impl)
            me.impl = self.impl - other.impl
            me.expl = self.expl - other.expl
            return me
        else:
            raise DataError("Type error: cannot subtract %s from %s" % (type(other), type(self)))

    def __add__(self, other):
        """
         Overloading the addition operator for rhs types

        Args:
            other (mesh.rhs_imex_field): rhs object to be added
        Raises:
            DataError: if other is not a rhs object
        Returns:
            mesh.rhs_imex_field: sum of caller and other values (self-other)
        """

        if isinstance(other, rhs_imex_dedalus_field):
            # always create new rhs_imex_field, since otherwise c = a + b changes a as well!
            me = rhs_imex_dedalus_field(self.impl)
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
             mesh.rhs_imex_field: copy of original values scaled by factor
        """

        if isinstance(other, float):
            # always create new rhs_imex_field
            me = rhs_imex_dedalus_field(self.impl)
            me.impl = other * self.impl
            me.expl = other * self.expl
            return me
        else:
            raise DataError("Type error: cannot multiply %s to %s" % (type(other), type(self)))

    # def apply_mat(self, A):
    #     """
    #     Matrix multiplication operator
    #
    #     Args:
    #         A: a matrix
    #
    #     Returns:
    #         mesh.rhs_imex_field: each component multiplied by the matrix A
    #     """
    #
    #     if not A.shape[1] == self.impl.values.shape[0]:
    #         raise DataError("ERROR: cannot apply operator %s to %s" % (A, self.impl))
    #     if not A.shape[1] == self.expl.values.shape[0]:
    #         raise DataError("ERROR: cannot apply operator %s to %s" % (A, self.expl))
    #
    #     me = rhs_imex_dedalus_field(self.domain)
    #     me.impl.values['g'] = A.dot(self.impl.values['g'])
    #     me.expl.values['g'] = A.dot(self.expl.values['g'])
    #
    #     return me
