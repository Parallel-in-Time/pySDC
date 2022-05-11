import cupy as cp
from pySDC.core.Errors import DataError

try:
    from mpi4py import MPI
except ImportError:
    MPI = None


class cupy_class(cp.ndarray):
    """
    cupy-based datatype
    """

    def __init__(cls, init, val=0.0, offset=0, buffer=None, strides=None, order=None):
        """
        Instantiates new datatype. This ensures that even when manipulating data, the result is still a mesh.

        Args:
            init: either another mesh or a tuple containing the dimensions and the dtype
            val: value to initialize

        Returns:
            obj of type mesh

        """
        if isinstance(init, cupy_class):
            obj = cp.ndarray.__init__(cls, shape=init.shape, dtype=init.dtype, strides=strides,
                                      order=order)  # TODO: not in cp.ndarray , buffer=buffer, offset=offset
            obj[:] = init[:]
        elif isinstance(init, tuple) and (init[1] is None or isinstance(init[1], MPI.Intracomm)) \
                and isinstance(init[2], cp.dtype):
            obj = cp.ndarray.__init__(cls, shape=init[0], dtype=init[2], strides=strides,
                                      order=order)  # TODO: not in cp.ndarray , buffer=buffer, offset=offset
            # obj = cp.full(shape=init[0], fill_value=val, dtype=init[2], order=order)
            # TODO: gibt es nicht für cupy
            # obj.fill(val)
            # print(init)
            print(type(obj))
            # print(obj.shape)
            obj[:] = cp.full(shape=init[0], fill_value=val, dtype=init[2], order=order)
        else:
            raise NotImplementedError(type(init))
        return obj

    def __array_ufunc__(self, ufunc, method, *inputs, out=None, **kwargs):
        """
        Overriding default ufunc, cf. https://numpy.org/doc/stable/user/basics.subclassing.html#array-ufunc-for-ufuncs
        for cupy: https://docs.cupy.dev/en/stable/user_guide/interoperability.html#numpy
        """
        args = []
        for _, input_ in enumerate(inputs):
            if isinstance(input_, cupy_class):
                args.append(input_.view(cp.ndarray))
            else:
                args.append(input_)
        results = super(cupy_class, self).__array_ufunc__(ufunc, method, *args, **kwargs).view(cupy_class)
        return results

    def __abs__(self):
        """
        Overloading the abs operator

        Returns:
            float: absolute maximum of all mesh values
        """
        # take absolute values of the mesh values
        local_absval = float(cp.amax(cp.ndarray.__abs__(self)))
        global_absval = local_absval
        return float(global_absval)


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
