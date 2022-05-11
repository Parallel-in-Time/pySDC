import cupy as cp
from pySDC.core.Errors import DataError

try:
    from mpi4py import MPI
except ImportError:
    MPI = None


class cupy_class:
    """
    cupy-based datatype
    """

    def __init__(self, init, val=0.0, offset=0, buffer=None, strides=None, order=None):
        """
        Instantiates new datatype. This ensures that even when manipulating data, the result is still a mesh.

        Args:
            init: either another mesh or a tuple containing the dimensions and the dtype
            val: value to initialize

        Attributes:
            values: cupy.ndarray

        """
        if isinstance(init, cupy_class):
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
        self.values[key] = value

    def __getitem__(self, item):
        return self.values[item]

    def flatten(self):
        return self.values.flatten()

    def __add__(self, other):
        if type(other) is cupy_class:
            new = cupy_class(other)
            new.values = other.values + self.values
        if type(other) is int or type(other) is float:
            new = cupy_class(self)
            new.values = other + self.values
        return new

    def __rmul__(self, other):
        new = None
        if type(other) is cupy_class:
            raise NotImplementedError("not implemendet to multiplicate to cupy_class obj")
        if type(other) is int or type(other) is float:
            new = cupy_class(self)
            new.values = other * self.values
        return new

    def __sub__(self, other):
        new = cupy_class(self)
        if type(other) is cupy_class:
            new.values = self.values - other.values
        if type(other) is int or type(other) is float:
            new = cupy_class(self)
            # TODO: check which one is korrekt
            # new.values = self.values - other
            # new.values = other - self.values
        return new

    def get(self):
        return self.values.get()



class imex_cupy_class(object):
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
                  or another imex_cupy_mesh object
            val (float): an initial number (default: 0.0)
        Raises:
            DataError: if init is none of the types above
        """

        if isinstance(init, type(self)):
            self.impl = cupy_class(init.impl)
            self.expl = cupy_class(init.expl)
        elif isinstance(init, tuple) and (init[1] is None or isinstance(init[1], MPI.Intracomm)) \
                and isinstance(init[2], cp.dtype):
            self.impl = cupy_class(init, val=val)
            self.expl = cupy_class(init, val=val)
        # something is wrong, if none of the ones above hit
        else:
            raise DataError('something went wrong during %s initialization' % type(self))


class comp2_cupy_class(object):
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
                  or another comp2_mesh object
        Raises:
            DataError: if init is none of the types above
        """

        if isinstance(init, type(self)):
            self.comp1 = cupy_class(init.comp1)
            self.comp2 = cupy_class(init.comp2)
        elif isinstance(init, tuple) and (init[1] is None or isinstance(init[1], MPI.Intracomm)) \
                and isinstance(init[2], cp.dtype):
            self.comp1 = cupy_class(init, val=val)
            self.comp2 = cupy_class(init, val=val)
        # something is wrong, if none of the ones above hit
        else:
            raise DataError('something went wrong during %s initialization' % type(self))
