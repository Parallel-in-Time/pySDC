import cupy as cp
from pySDC.core.errors import DataError

try:
    from mpi4py import MPI
except ImportError:
    MPI = None


class cupy_mesh(cp.ndarray):
    """
    CuPy-based datatype for serial or parallel meshes.
    """

    def __new__(cls, init, val=0.0, offset=0, buffer=None, strides=None, order=None):
        """
        Instantiates new datatype. This ensures that even when manipulating data, the result is still a mesh.

        Args:
            init: either another mesh or a tuple containing the dimensions, the communicator and the dtype
            val: value to initialize

        Returns:
            obj of type mesh

        """
        if isinstance(init, cupy_mesh):
            obj = cp.ndarray.__new__(cls, shape=init.shape, dtype=init.dtype, strides=strides, order=order)
            obj[:] = init[:]
            obj._comm = init._comm
        elif (
            isinstance(init, tuple)
            and (init[1] is None or isinstance(init[1], MPI.Intracomm))
            and isinstance(init[2], cp.dtype)
        ):
            obj = cp.ndarray.__new__(cls, init[0], dtype=init[2], strides=strides, order=order)
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
        for _, input_ in enumerate(inputs):
            if isinstance(input_, cupy_mesh):
                args.append(input_.view(cp.ndarray))
                comm = input_.comm
            else:
                args.append(input_)
        results = super(cupy_mesh, self).__array_ufunc__(ufunc, method, *args, **kwargs).view(cupy_mesh)
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
        local_absval = float(cp.amax(cp.ndarray.__abs__(self)))

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


class CuPyMultiComponentMesh(cupy_mesh):
    r"""
    Generic mesh with multiple components.

    To make a specific multi-component mesh, derive from this class and list the components as strings in the class
    attribute ``components``. An example:

    ```
    class imex_cupy_mesh(CuPyMultiComponentMesh):
        components = ['impl', 'expl']
    ```

    Instantiating such a mesh will expand the mesh along an added first dimension for each component and allow access
    to the components with ``.``. Continuing the above example:

    ```
    init = ((100,), None, numpy.dtype('d'))
    f = imex_cupy_mesh(init)
    f.shape  # (2, 100)
    f.expl.shape  # (100,)
    ```

    Note that the components are not attributes of the mesh: ``"expl" in dir(f)`` will return False! Rather, the
    components are handled in ``__getattr__``. This function is called if an attribute is not found and returns a view
    on to the component if appropriate. Importantly, this means that you cannot name a component like something that
    is already an attribute of ``cupy_mesh`` or ``cupy.ndarray`` because this will not result in calls to ``__getattr__``.

    There are a couple more things to keep in mind:
     - Because a ``CuPyMultiComponentMesh`` is just a ``cupy.ndarray`` with one more dimension, all components must have
       the same shape.
     - You can use the entire ``CuPyMultiComponentMesh`` like a ``cupy.ndarray`` in operations that accept arrays, but make
       sure that you really want to apply the same operation on all components if you do.
     - If you omit the assignment operator ``[:]`` during assignment, you will not change the mesh at all. Omitting this
       leads to all kinds of trouble throughout the code. But here you really cannot get away without.
    """

    components = []

    def __new__(cls, init, *args, **kwargs):
        if isinstance(init, tuple):
            shape = (init[0],) if type(init[0]) is int else init[0]
            obj = super().__new__(cls, ((len(cls.components), *shape), *init[1:]), *args, **kwargs)
        else:
            obj = super().__new__(cls, init, *args, **kwargs)

        return obj

    def __getattr__(self, name):
        if name in self.components:
            if self.shape[0] == len(self.components):
                return self[self.components.index(name)].view(cupy_mesh)
            else:
                raise AttributeError(f'Cannot access {name!r} in {type(self)!r} because the shape is unexpected.')
        else:
            raise AttributeError(f"{type(self)!r} does not have attribute {name!r}!")


class imex_cupy_mesh(CuPyMultiComponentMesh):
    components = ['impl', 'expl']


class comp2_cupy_mesh(CuPyMultiComponentMesh):
    components = ['comp1', 'comp2']
