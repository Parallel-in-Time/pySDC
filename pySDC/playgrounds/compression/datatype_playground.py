from pySDC.implementations.datatype_classes.mesh import mesh
import numpy as np
from pySDC.projects.compression.CRAM_Manager import CRAM_Manager
from pySDC.core.Errors import DataError

# class compressed_mesh(np.ndarray):
#     def __new__(cls, init, val=0.0, varName="---", **kwargs):
#         if isinstance(init, compressed_mesh):
#             obj = np.ndarray.__new__(cls, shape=init.shape, dtype=init.dtype, **kwargs)
#             obj._comm = init._comm
#             obj.varName = varName
#             obj[:] = init[:]
#             manager.registerVar(
#                 varName,
#                 init,
#                 init.dtype,
#                 numVectors=1,
#                 errBoundMode="ABS",
#                 compType="sz",
#                 errBound=1e-5,
#             )

#         elif (
#             isinstance(init, tuple)
#             and (init[1] is None or isinstance(init[1], MPI.Intracomm))
#             and isinstance(init[2], np.dtype)
#         ):
#             obj = np.ndarray.__new__(cls, init[0], dtype=init[2], **kwargs)
#             obj.fill(val)
#             obj._comm = init[1]
#             obj.varName = varName
#             manager.registerVar(
#                 varName,
#                 init,
#                 init[2],
#                 numVectors=1,
#                 errBoundMode="ABS",
#                 compType="sz3",
#                 errBound=1e-5,
#             )

#         else:
#             raise NotImplementedError(type(init))
#         return obj

#     # compressor manager, setter and getter
#     # Operations: Assignment, add, subtract, scale

#     def __getitem__(self, key):
#         array = manager.decompress(self.varName, 0)
#         return array.__getitem__(key)
#         # print("Get: ", key)# super().__getitem__(key))
#         # if isinstance(key, slice):
#         #    array = manager.decompress(self.varName,0)
#         #    return array.__getitem__(key)
#         #    indices = range(*key.indices(len(self.list)))
#         #    return [self.list[i] for i in indices]

#         # return super().__getitem__(key)
#         # return manager.decompress(self.varName,0)

#     def __setitem__(self, key, newvalue):
#         print("SET: ", key, newvalue)
#         array = manager.decompress(self.varName, 0)
#         array.__setitem__(key, newvalue)
#         manager.compress(array, self.varName, 0)

#         # if isinstance(key, slice):
#         #     if newvalue == 1:
#         #         array = np.ones(self.init.shape)*newvalue
#         #         self.manager.compress(array, self.varName,0)
#         #     else:
#         #         array = self.manager.decompress(self.varName,0)
#         #         array.__setitem__(key, newvalue)
#         #         self.manager.compress(array, self.varName,0)
#         # else:
#         #     array = self.manager.decompress(self.varName,0)
#         #     array.__setitem__(key, newvalue)
#         # #super().__setitem__(key, newvalue)
#         # #self.manager.compress(newvalue, self.varName,0)

#     def __array_finalize__(self, obj):
#         """
#         Finalizing the datatype. Without this, new datatypes do not 'inherit' the communicator.
#         """
#         if obj is None:
#             return
#         self._comm = getattr(obj, "_comm", None)
#         self.varName = getattr(obj, "varName", None)

#     def __array_ufunc__(self, ufunc, method, *inputs, out=None, **kwargs):
#         """
#         Overriding default ufunc, cf. https://numpy.org/doc/stable/user/basics.subclassing.html#array-ufunc-for-ufuncs
#         """
#         args = []
#         comm = None
#         varName = None
#         print("Inputs: ", inputs)
#         for _, input_ in enumerate(inputs):
#             if isinstance(input_, compressed_mesh):
#                 array = manager.decompress(input_.varName, 0)
#                 args.append(array.view(np.ndarray))
#                 comm = input_._comm
#                 varName = input_.varName
#             else:
#                 args.append(input_)
#         results = super(compressed_mesh, self).__array_ufunc__(ufunc, method, *args, **kwargs).view(np.ndarray)
#         if not method == "reduce":
#             cprss_array = compressed_mesh(input_, varName=str(name + 1))
#             cprss_array._comm = comm
#             cprss_array[:] = results
#         return cprss_array

#     # def __add__(self, x)
#     #    a = manager.decompress(self.varName,0)
#     #    b = manager.decompress(x.varName,0)

#     #    c = a + b


class compressed_meshV2(object):
    """
    Mesh data type with arbitrary dimensions

    This data type can be used whenever structured data with a single unknown per point in space is required

    Attributes:
        values (np.ndarray): contains the ndarray of the values
    """

    manager = CRAM_Manager("ABS", "sz", 1)

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
        if isinstance(init, compressed_meshV2):
            values = self.manager.decompress(init.name, 0)  # TODO: Modify manager to copy compressed buffer
            self.manager.registerVar(
                self.name,
                values.shape,
                values.dtype,
                numVectors=1,
                errBoundMode="ABS",
                compType="sz3",
                errBound=1e-5,
            )
            self.manager.compress(values.copy(), self.name, 0)
        # if init is a number or a tuple of numbers, create mesh object with val as initial value
        elif isinstance(init, tuple) or isinstance(init, int):
            self.manager.registerVar(
                self.name,
                init[0],
                init[2],
                numVectors=1,
                errBoundMode="ABS",
                compType="sz3",
                errBound=1e-5,
            )
            self.manager.compress(np.full(init[0], fill_value=val), self.name, 0)
        # something is wrong, if none of the ones above hit
        else:
            raise DataError("something went wrong during %s initialization" % type(self))

    def __del__(self):
        print("Delete" + " " + self.name)
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

        if isinstance(other, compressed_meshV2):
            # always create new mesh, since otherwise c = a + b changes a as well!
            me = compressed_meshV2(self)
            values = self.manager.decompress(self.name, 0)
            ov = self.manager.decompress(other.name, 0)
            self.manager.compress(values + ov, me.name, 0)
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

        if isinstance(other, compressed_meshV2):
            # always create new mesh, since otherwise c = a - b changes a as well!
            me = compressed_meshV2(self)
            values = self.manager.decompress(self.name, 0)
            ov = self.manager.decompress(other.name, 0)
            self.manager.compress(values - ov, me.name, 0)
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
            values = self.manager.decompress(self.name, 0)
            me = compressed_meshV2(self)
            self.manager.compress(values * other, me.name, 0)
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
        values = self.manager.decompress(self.name, 0)
        absval = abs(values)

        # return maximum
        return np.amax(absval)

    def __setitem__(self, key, newvalue):
        print("SET: ", key, newvalue)
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


N = 30
init = (N, None, np.dtype("float64"))
a = compressed_meshV2(init)
# a1 = compressed_mesh(init, varName='A')
b = compressed_meshV2(init)
c = compressed_meshV2(init)
a[:] = 1
b[:] = 2
print(a)
print(b)
c[:] = a + b
print(type(a + b))
print(type(c))
print(c[:])
print(c.name)
print(a.manager)
del a
print(b.manager)
