# from pySDC.implementations.datatype_classes.compressed_mesh import compressed_mesh
import numpy as np
from pySDC.projects.compression.CRAM_Manager import CRAM_Manager
from pySDC.core.Errors import DataError


class compressed_mesh(object):
    """
    Mesh data type with arbitrary dimensions

    This data type can be used whenever structured data with a single unknown per point in space is required

    Attributes:
        values (np.ndarray): contains the ndarray of the values
    """

    manager = CRAM_Manager("ABS", "sz3", 1e-5)

    def __init__(self, init=None, val=0.0):
        """
        Initialization routine

        Args:
            init: can either be a tuple (one int per dimension) or a number (if only one dimension is requested)
                  or another mesh object
            val: initial value (default: None)
        Raises:
            DataError: if init is none of the types above
        """
        self.name = str(self.manager.name + 1)
        self.manager.name += 1
        # if init is another mesh, do a copy (init by copy)
        if isinstance(init, compressed_mesh):
            values = self.manager.decompress(
                init.name, 0
            )  # TODO: Modify manager to copy compressed buffer
            self.manager.registerVar(
                self.name,
                values.shape,
                values.dtype,
                # numVectors=1,
                # errBoundMode="ABS",
                # compType="sz3",
                # errBound=self.manager.errBound,
            )
            self.manager.compress(values.copy(), self.name, 0)
        # if init is a number or a tuple of numbers, create mesh object with val as initial value
        elif isinstance(init, tuple) or isinstance(init, int):
            self.manager.registerVar(
                self.name,
                init[0],
                init[2],
                # numVectors=1,
                # errBoundMode="ABS",
                # compType="sz3",
                # errBound=self.manager.errBound,
            )
            self.manager.compress(
                np.full(init[0], fill_value=val, dtype=np.dtype("float64")),
                self.name,
                0,
            )
        # something is wrong, if none of the ones above hit
        else:
            raise DataError(
                "something went wrong during %s initialization" % type(self)
            )

    def __del__(self):
        # print('Delete'+' ' +self.name)
        self.manager.remove(self.name, 0)

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

        me = compressed_mesh(self)
        values = self.manager.decompress(self.name, 0)

        if isinstance(other, compressed_mesh):
            # always create new mesh, since otherwise c = a - b changes a as well!
            ov = self.manager.decompress(other.name, 0)
        # else:
        #     raise DataError(
        #         "Type error: cannot subtract %s from %s" % (type(other), type(self))
        #     )
        else:
            ov = other
        self.manager.compress(values + ov, me.name, 0)
        return me

    def __radd__(self, other):
        return self.__add__(other)

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
        me = compressed_mesh(self)
        values = self.manager.decompress(self.name, 0)

        if isinstance(other, compressed_mesh):
            # always create new mesh, since otherwise c = a - b changes a as well!
            ov = self.manager.decompress(other.name, 0)
        # else:
        #     raise DataError(
        #         "Type error: cannot subtract %s from %s" % (type(other), type(self))
        #     )
        else:
            ov = other
        self.manager.compress(values - ov, me.name, 0)
        return me

    def __rsub__(self, other):
        me = compressed_mesh(self)
        values = self.manager.decompress(self.name, 0)

        if isinstance(other, compressed_mesh):
            # always create new mesh, since otherwise c = a - b changes a as well!
            ov = self.manager.decompress(other.name, 0)
        # else:
        #     raise DataError(
        #         "Type error: cannot subtract %s from %s" % (type(other), type(self))
        #     )
        else:
            ov = other
        self.manager.compress(ov - values, me.name, 0)
        return me

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
        me = compressed_mesh(self)
        values = self.manager.decompress(self.name, 0)

        if isinstance(other, compressed_mesh):
            # always create new mesh, since otherwise c = a - b changes a as well!
            ov = self.manager.decompress(other.name, 0)
        # else:
        #     raise DataError(
        #         "Type error: cannot subtract %s from %s" % (type(other), type(self))
        #     )
        else:
            ov = other
        self.manager.compress(ov * values, me.name, 0)
        return me

    def __mul__(self, other):
        return self.__rmul__(other)

    def __abs__(self):
        """
        Overloading the abs operator for mesh types

        Returns:
            float: absolute maximum of all mesh values
        """

        # take absolute values of the mesh values
        values = self.manager.decompress(self.name, 0)
        absval = abs(values)

        # return maximum
        return np.amax(absval)

    def __setitem__(self, key, newvalue):
        # print("SET: ", key, newvalue)
        if type(newvalue) == type(self):  # Assigning compressed mesh
            arr_temp = self.manager.decompress(newvalue.name, 0)
            self.manager.compress(arr_temp, self.name, 0)
        else:
            array = self.manager.decompress(self.name, 0)
            array.__setitem__(key, newvalue)
            self.manager.compress(array, self.name, 0)

    def __getitem__(self, key):
        array = self.manager.decompress(self.name, 0)
        return array.__getitem__(key)

    def __str__(self):
        return str(self[:])

    def flatten(self):
        return self.manager.decompress(self.name, 0).flatten()


'''
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
        return comm.Issend(self.values[:], dest=dest, tag=tag)

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
        return comm.Irecv(self.values[:], source=source, tag=tag)

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
'''


class imex_mesh_compressed(object):
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

        # if init is another rhs_imex_mesh, do a copy (init by copy)
        if isinstance(init, type(self)):
            self.impl = compressed_mesh(init.impl)
            self.expl = compressed_mesh(init.expl)
        # if init is a number or a tuple of numbers, create compressed_mesh object with None as initial value
        elif isinstance(init, tuple) or isinstance(init, int):
            self.impl = compressed_mesh(init, val=val)
            self.expl = compressed_mesh(init, val=val)
        # something is wrong, if none of the ones above hit
        else:
            raise DataError(
                "something went wrong during %s initialization" % type(self)
            )

    # def __sub__(self, other):
    #     """
    #     Overloading the subtraction operator for rhs types

    #     Args:
    #         other (compressed_mesh.rhs_imex_compressed_mesh): rhs object to be subtracted
    #     Raises:
    #         DataError: if other is not a rhs object
    #     Returns:
    #         compressed_mesh.rhs_imex_compressed_mesh: differences between caller and other values (self-other)
    #     """

    #     if isinstance(other, rhs_imex_compressed_mesh):
    #         # always create new rhs_imex_compressed_mesh, since otherwise c = a - b changes a as well!
    #         me = rhs_imex_compressed_mesh(self)
    #         me.impl.values = self.impl.values - other.impl.values
    #         me.expl.values = self.expl.values - other.expl.values
    #         return me
    #     else:
    #         raise DataError("Type error: cannot subtract %s from %s" % (type(other), type(self)))

    # def __add__(self, other):
    #     """
    #      Overloading the addition operator for rhs types

    #     Args:
    #         other (compressed_mesh.rhs_imex_compressed_mesh): rhs object to be added
    #     Raises:
    #         DataError: if other is not a rhs object
    #     Returns:
    #         compressed_mesh.rhs_imex_compressed_mesh: sum of caller and other values (self-other)
    #     """

    #     if isinstance(other, rhs_imex_compressed_mesh):
    #         # always create new rhs_imex_compressed_mesh, since otherwise c = a + b changes a as well!
    #         me = rhs_imex_compressed_mesh(self)
    #         me.impl.values = self.impl.values + other.impl.values
    #         me.expl.values = self.expl.values + other.expl.values
    #         return me
    #     else:
    #         raise DataError("Type error: cannot add %s to %s" % (type(other), type(self)))

    # def __rmul__(self, other):
    #     """
    #     Overloading the right multiply by factor operator for compressed_mesh types

    #     Args:
    #         other (float): factor
    #     Raises:
    #         DataError: is other is not a float
    #     Returns:
    #          compressed_mesh.rhs_imex_compressed_mesh: copy of original values scaled by factor
    #     """

    #     if isinstance(other, float):
    #         # always create new rhs_imex_compressed_mesh
    #         me = rhs_imex_compressed_mesh(self)
    #         me.impl.values = other * self.impl.values
    #         me.expl.values = other * self.expl.values
    #         return me
    #     else:
    #         raise DataError("Type error: cannot multiply %s to %s" % (type(other), type(self)))

    # def apply_mat(self, A):
    #     """
    #     Matrix multiplication operator

    #     Args:
    #         A: a matrix

    #     Returns:
    #         compressed_mesh.rhs_imex_compressed_mesh: each component multiplied by the matrix A
    #     """

    #     if not A.shape[1] == self.impl.values.shape[0]:
    #         raise DataError("ERROR: cannot apply operator %s to %s" % (A, self.impl))
    #     if not A.shape[1] == self.expl.values.shape[0]:
    #         raise DataError("ERROR: cannot apply operator %s to %s" % (A, self.expl))

    #     me = rhs_imex_compressed_mesh(A.shape[1])
    #     me.impl.values = A.dot(self.impl.values)
    #     me.expl.values = A.dot(self.expl.values)

    #     return me
