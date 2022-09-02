import cupy as cp
from pySDC.core.Errors import DataError

try:
    from mpi4py import MPI
except ImportError:
    MPI = None


class cupy_mesh:
    """
    cupy-based datatype
    """

    def __init__(self, init, val=0.0, offset=0, buffer=None, strides=None, order=None):
        """
        Instantiates new datatype. This ensures that even when manipulating data, the result is still a cupy_mesh.

        Args:
            init: either another mesh or a tuple containing the dimensions and the dtype
            val: value to initialize

        Attributes:
            values: cupy.ndarray

        """
        if isinstance(init, cupy_mesh):
            self.values = cp.ndarray(shape=init.values.shape, dtype=init.values.dtype, strides=strides, order=order)
            self.values[:] = init.values.copy()
        elif isinstance(init, tuple) and (init[1] is None or isinstance(init[1], MPI.Intracomm)) \
                and isinstance(init[2], cp.dtype):
            self.values = cp.ndarray(shape=init[0], dtype=init[2], strides=strides, order=order)
            self.values[:] = cp.full(shape=init[0], fill_value=val, dtype=init[2], order=order)
        else:
            raise NotImplementedError(type(init))

    def __abs__(self):
        """
        Overloading the abs operator

        Returns:
            float: absolute maximum of all mesh values
        """
        # take absolute values of the mesh values
        local_absval = float(cp.amax(cp.ndarray.__abs__(self.values)))
        global_absval = local_absval
        return float(global_absval)

    def __setitem__(self, key, value):
        """
        Overloading the setitem operator
        """
        self.values[key] = value

    def __getitem__(self, item):
        """
        Overloading the getitem operator
        """
        return self.values[item]

    def flatten(self):
        """
        Overloading the flatten operator
        Returns:
             cupy.ndarray
        """
        return self.values.flatten()

    def __add__(self, other):
        """
        Overloading the add operator
        Returns:
            new: cupy_mesh
        """
        if type(other) is cupy_mesh:
            new = cupy_mesh(other)
            new.values = other.values + self.values
        if type(other) is int or type(other) is float:
            new = cupy_mesh(self)
            new.values = other + self.values
        return new

    def __rmul__(self, other):
        """
        Overloading the rmul operator
        Returns:
            new: cupy_mesh
        """
        new = None
        if type(other) is cupy_mesh:
            raise NotImplementedError("not implemendet to multiplicate to cupy_class obj")
        if type(other) is int or type(other) is float:
            new = cupy_mesh(self)
            new.values = other * self.values
        return new

    def __sub__(self, other):
        """
        Overloading the sub operator
        Returns:
            new: cupy_mesh
        """
        new = cupy_mesh(self)
        if type(other) is cupy_mesh:
            new.values = self.values - other.values
        if type(other) is int or type(other) is float:
            new = cupy_mesh(self)
            new.values = self.values - other
        return new

    def get(self):
        """
        Overloading the get operator from cupy.ndarray
        Returns:
            numpy.ndarray
        """
        return self.values.get()


class imex_cupy_mesh(object):
    """
    RHS data type for cupy_meshes with implicit and explicit components

    This data type can be used to have RHS with 2 components (here implicit and explicit)

    Attributes:
        impl (cupy_mesh.cupy_mesh): implicit part
        expl (cupy_mesh.cupy_mesh): explicit part
    """

    def __init__(self, init, val=0.0):
        """
        Initialization routine

        Args:
            init: can either be a tuple (one int per dimension) or a number (if only one dimension is requested)
                  or another imex_cupy_mesh object
            val (float): an initial number (default: 0.0)
        Raises:
            DataError: if init is none of the types above
        """

        if isinstance(init, type(self)):
            self.impl = cupy_mesh(init.impl)
            self.expl = cupy_mesh(init.expl)
        elif isinstance(init, tuple) and (init[1] is None or isinstance(init[1], MPI.Intracomm)) \
                and isinstance(init[2], cp.dtype):
            self.impl = cupy_mesh(init, val=val)
            self.expl = cupy_mesh(init, val=val)
        # something is wrong, if none of the ones above hit
        else:
            raise DataError('something went wrong during %s initialization' % type(self))


class comp2_cupy_mesh(object):
    """
    RHS data type for cupy_meshes with 2 components

    Attributes:
        comp1 (cupy_mesh.cupy_mesh): first part
        comp2 (cupy_mesh.cupy_mesh): second part
    """

    def __init__(self, init, val=0.0):
        """
        Initialization routine

        Args:
            init: can either be a tuple (one int per dimension) or a number (if only one dimension is requested)
                  or another comp2_mesh object
        Raises:
            DataError: if init is none of the types above
        """

        if isinstance(init, type(self)):
            self.comp1 = cupy_mesh(init.comp1)
            self.comp2 = cupy_mesh(init.comp2)
        elif isinstance(init, tuple) and (init[1] is None or isinstance(init[1], MPI.Intracomm)) \
                and isinstance(init[2], cp.dtype):
            self.comp1 = cupy_mesh(init, val=val)
            self.comp2 = cupy_mesh(init, val=val)
        # something is wrong, if none of the ones above hit
        else:
            raise DataError('something went wrong during %s initialization' % type(self))
