import numpy as np

try:
    # TODO : mpi4py cannot be imported before dolfin when using fenics mesh
    # see https://github.com/Parallel-in-Time/pySDC/pull/285#discussion_r1145850590
    # This should be dealt with at some point
    from mpi4py import MPI
except ImportError:
    MPI = None


class mesh(np.ndarray):
    """
    Numpy-based datatype for serial or parallel meshes.
    Can include a communicator and expects a dtype to allow complex data.

    Attributes:
        comm: MPI communicator or None
    """

    comm = None
    xp = np

    def __new__(cls, init, val=0.0, **kwargs):
        """
        Instantiates new datatype. This ensures that even when manipulating data, the result is still a mesh.

        Args:
            init: either another mesh or a tuple containing the dimensions, the communicator and the dtype
            val: value to initialize

        Returns:
            obj of type mesh

        """
        if isinstance(init, mesh):
            obj = np.ndarray.__new__(cls, shape=init.shape, dtype=init.dtype, **kwargs)
            obj[:] = init[:]
        elif (
            isinstance(init, tuple)
            and (init[1] is None or isinstance(init[1], MPI.Intracomm))
            and isinstance(init[2], np.dtype)
        ):
            obj = np.ndarray.__new__(cls, init[0], dtype=init[2], **kwargs)
            obj.fill(val)
            cls.comm = init[1]
        else:
            raise NotImplementedError(type(init))
        return obj

    def __array_ufunc__(self, ufunc, method, *inputs, out=None, **kwargs):
        """
        Overriding default ufunc, cf. https://numpy.org/doc/stable/user/basics.subclassing.html#array-ufunc-for-ufuncs
        """
        args = []
        for _, input_ in enumerate(inputs):
            if isinstance(input_, mesh):
                args.append(input_.view(np.ndarray))
            else:
                args.append(input_)

        results = super().__array_ufunc__(ufunc, method, *args, **kwargs).view(type(self))
        return results

    def __abs__(self):
        """
        Overloading the abs operator

        Returns:
            float: absolute maximum of all mesh values
        """
        # take absolute values of the mesh values
        local_absval = float(np.max(np.ndarray.__abs__(self)))

        if self.comm is not None:
            if self.comm.size > 1:
                global_absval = self.comm.allreduce(sendobj=local_absval, op=MPI.MAX)
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


def _component_property(index, name, base):
    """
    Make a property that reads and writes the ``index``-th component of a multi-component mesh.

    Args:
        index (int): position of the component along the first axis
        name (str): name of the component, used in error messages
        base: the single-component datatype a component is a view of

    Returns:
        property: gives a view on the component, and writes into it on assignment
    """

    def getter(self):
        if self.shape[0] != len(self.components):
            raise AttributeError(f'Cannot access {name!r} in {type(self)!r} because the shape is unexpected.')
        return self[index].view(base)

    def setter(self, value):
        getter(self)[:] = value

    return property(getter, setter, doc=f'View on the {name!r} component of the mesh')


class MultiComponentMeshMixin:
    r"""
    Generic mesh with multiple components.

    Mix this into a mesh datatype to obtain the multi-component version of it, as ``MultiComponentMesh`` does for
    ``mesh`` and ``CuPyMultiComponentMesh`` does for ``cupy_mesh``. To make a specific multi-component mesh, derive
    from one of those and list the components as strings in the class attribute ``components``. An example:

    ```
    class imex_mesh(MultiComponentMesh):
        components = ['impl', 'expl']
    ```

    Instantiating such a mesh will expand the mesh along an added first dimension for each component and allow access
    to the components with ``.``. Continuing the above example:

    ```
    init = ((100,), None, numpy.dtype('d'))
    f = imex_mesh(init)
    f.shape  # (2, 100)
    f.expl.shape  # (100,)
    ```

    The components are properties, generated when the subclass is created. Both ``f.expl[:] = ...`` and
    ``f.expl = ...`` write into the mesh; the component is never replaced by an unrelated object. Because the
    properties live on the class, you cannot name a component like something that is already an attribute of the
    underlying mesh datatype -- doing so raises an ``AttributeError`` when the class is created rather than silently
    shadowing the component.

    There are a couple more things to keep in mind:
     - Because a multi-component mesh is just an array with one more dimension, all components must have the same
       shape.
     - You can use the entire mesh like an array in operations that accept arrays, but make sure that you really want
       to apply the same operation on all components if you do.
    """

    components = []

    def __init_subclass__(cls, **kwargs):
        """
        Turn the names listed in ``components`` into properties giving a view on the corresponding slice of the mesh.
        """
        super().__init_subclass__(**kwargs)

        # a single component is an instance of the datatype this multi-component mesh is built on
        base = next(c for c in cls.__mro__ if not issubclass(c, MultiComponentMeshMixin))

        for index, name in enumerate(cls.components):
            if hasattr(base, name):
                raise AttributeError(
                    f'Cannot use {name!r} as a component of {cls.__name__} because it is already an attribute of '
                    f'{base.__name__}!'
                )
            setattr(cls, name, _component_property(index, name, base))

    def __new__(cls, init, *args, **kwargs):
        if isinstance(init, tuple):
            shape = (init[0],) if type(init[0]) is int else init[0]
            obj = super().__new__(cls, ((len(cls.components), *shape), *init[1:]), *args, **kwargs)
        else:
            obj = super().__new__(cls, init, *args, **kwargs)

        return obj


class MultiComponentMesh(MultiComponentMeshMixin, mesh):
    """Numpy-based mesh with multiple components, see ``MultiComponentMeshMixin``."""


class imex_mesh(MultiComponentMesh):
    components = ['impl', 'expl']


class comp2_mesh(MultiComponentMesh):
    components = ['comp1', 'comp2']
