import numpy as np

from pySDC.core.Errors import DataError

try:
    from mpi4py import MPI
except ImportError:
    MPI = None


class parallel_mesh(np.ndarray):
    """
    Numpy-based datatype for serial or parallel meshes.
    Can include a communicator and expects a dtype to allow complex data.

    Attributes:
        _comm: MPI communicator or None
    """

    def __new__(cls, init, val=0.0, offset=0, buffer=None, strides=None, order=None):
        """
        Instantiates new datatype. This ensures that even when manipulating data, the result is still a parallel_mesh.

        Args:
            init: either another parallel_mesh or a tuple containing the dimensions, the communicator and the dtype
            val: value to initialize

        Returns:
            obj of type parallel_mesh

        """
        if isinstance(init, parallel_mesh):
            obj = np.ndarray.__new__(cls, shape=init.shape, dtype=init.dtype, buffer=buffer, offset=offset,
                                     strides=strides, order=order)
            obj[:] = init[:]
            obj._comm = init._comm
        elif isinstance(init, tuple) and (init[1] is None or isinstance(init[1], MPI.Intracomm)) \
                and isinstance(init[2], np.dtype):
            obj = np.ndarray.__new__(cls, init[0], dtype=init[2], buffer=buffer, offset=offset,
                                     strides=strides, order=order)
            obj.fill(val)
            obj._comm = init[1]
        else:
            raise NotImplementedError(type(init))
        return obj

    @property
    def comm(self):
        """
        Getter for the communicator
        """
        return self._comm

    def __array_finalize__(self, obj):
        """
        Finalizing the datatype. Without this, new datatypes do not 'inherit' the communicator.
        """
        if obj is None:
            return
        self._comm = getattr(obj, '_comm', None)

    def __array_ufunc__(self, ufunc, method, *inputs, out=None, **kwargs):
        """
        Overriding default ufunc, cf. https://numpy.org/doc/stable/user/basics.subclassing.html#array-ufunc-for-ufuncs
        """
        args = []
        comm = None
        for i, input_ in enumerate(inputs):
            if isinstance(input_, parallel_mesh):
                args.append(input_.view(np.ndarray))
                comm = input_.comm
            else:
                args.append(input_)
        results = super(parallel_mesh, self).__array_ufunc__(ufunc, method, *args, **kwargs).view(parallel_mesh)
        if not method == 'reduce':
            results._comm = comm
        return results

    def __abs__(self):
        """
        Overloading the abs operator

        Returns:
            float: absolute maximum of all mesh values
        """
        # take absolute values of the mesh values
        local_absval = float(np.amax(np.ndarray.__abs__(self)))

        if self.comm is not None:
            if self.comm.Get_size() > 1:
                global_absval = 0.0
                global_absval = max(self.comm.allreduce(sendobj=local_absval, op=MPI.MAX), global_absval)
            else:
                global_absval = local_absval
        else:
            global_absval = local_absval

        return float(global_absval)

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
        return comm.Issend(self[:], dest=dest, tag=tag)

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
        return comm.Irecv(self[:], source=source, tag=tag)

    def bcast(self, root=None, comm=None):
        """
        Routine for broadcasting values

        Args:
            root (int): process with value to broadcast
            comm: communicator

        Returns:
            broadcasted values
        """
        comm.Bcast(self[:], root=root)
        return self


class parallel_imex_mesh(object):
    """
    RHS data type for meshes with implicit and explicit components

    This data type can be used to have RHS with 2 components (here implicit and explicit)

    Attributes:
        impl (parallel_mesh.parallel_mesh): implicit part
        expl (parallel_mesh.parallel_mesh): explicit part
    """

    def __init__(self, init, val=0.0):
        """
        Initialization routine

        Args:
            init: can either be a tuple (one int per dimension) or a number (if only one dimension is requested)
                  or another parallel_imex_mesh object
            val (float): an initial number (default: 0.0)
        Raises:
            DataError: if init is none of the types above
        """

        if isinstance(init, type(self)):
            self.impl = parallel_mesh(init.impl)
            self.expl = parallel_mesh(init.expl)
        elif isinstance(init, tuple) and (init[1] is None or isinstance(init[1], MPI.Intracomm)) \
                and isinstance(init[2], np.dtype):
            self.impl = parallel_mesh(init, val=val)
            self.expl = parallel_mesh(init, val=val)
        # something is wrong, if none of the ones above hit
        else:
            raise DataError('something went wrong during %s initialization' % type(self))


class parallel_comp2_mesh(object):
    """
    RHS data type for meshes with 2 components

    Attributes:
        comp1 (parallel_mesh.parallel_mesh): first part
        comp2 (parallel_mesh.parallel_mesh): second part
    """

    def __init__(self, init, val=0.0):
        """
        Initialization routine

        Args:
            init: can either be a tuple (one int per dimension) or a number (if only one dimension is requested)
                  or another parallel_comp2_mesh object
        Raises:
            DataError: if init is none of the types above
        """

        if isinstance(init, type(self)):
            self.comp1 = parallel_mesh(init.comp1)
            self.comp2 = parallel_mesh(init.comp2)
        elif isinstance(init, tuple) and (init[1] is None or isinstance(init[1], MPI.Intracomm)) \
                and isinstance(init[2], np.dtype):
            self.comp1 = parallel_mesh(init, val=val)
            self.comp2 = parallel_mesh(init, val=val)
        # something is wrong, if none of the ones above hit
        else:
            raise DataError('something went wrong during %s initialization' % type(self))
