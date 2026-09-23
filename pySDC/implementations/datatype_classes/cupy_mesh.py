import cupy as cp

try:
    from mpi4py import MPI
except ImportError:
    MPI = None

from pySDC.implementations.datatype_classes.mesh import MultiComponentMeshMixin

try:
    from pySDC.helpers.NCCL_communicator import NCCLComm
except ImportError:
    NCCLComm = None


class cupy_mesh(cp.ndarray):
    """
    CuPy-based datatype for serial or parallel meshes.
    """

    comm = None
    xp = cp

    def __new__(cls, init, val=0.0, **kwargs):
        """
        Instantiates new datatype. This ensures that even when manipulating data, the result is still a mesh.

        Args:
            init: either another mesh or a tuple containing the dimensions, the communicator and the dtype
            val: value to initialize

        Returns:
            obj of type mesh

        """
        if isinstance(init, cupy_mesh):
            obj = cp.ndarray.__new__(cls, shape=init.shape, dtype=init.dtype, **kwargs)
            obj[:] = init[:]
        elif (
            isinstance(init, tuple)
            and (init[1] is None or isinstance(init[1], MPI.Intracomm) or isinstance(init[1], NCCLComm))
            and isinstance(init[2], cp.dtype)
        ):
            obj = cp.ndarray.__new__(cls, init[0], dtype=init[2], **kwargs)
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
            if isinstance(input_, cupy_mesh):
                args.append(input_.view(cp.ndarray))
            else:
                args.append(input_)
        results = super(cupy_mesh, self).__array_ufunc__(ufunc, method, *args, **kwargs).view(type(self))
        return results

    def __abs__(self):
        """
        Overloading the abs operator

        Returns:
            float: absolute maximum of all mesh values
        """
        # take absolute values of the mesh values
        local_absval = cp.max(cp.ndarray.__abs__(self))

        if self.comm is not None:
            if self.comm.Get_size() > 1:
                global_absval = local_absval * 0
                if isinstance(self.comm, NCCLComm):
                    self.comm.Allreduce(sendbuf=local_absval, recvbuf=global_absval, op=MPI.MAX)
                else:
                    global_absval = self.comm.allreduce(sendobj=float(local_absval), op=MPI.MAX)
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


class CuPyMultiComponentMesh(MultiComponentMeshMixin, cupy_mesh):
    """CuPy-based mesh with multiple components, see ``MultiComponentMeshMixin``."""


class imex_cupy_mesh(CuPyMultiComponentMesh):
    components = ['impl', 'expl']


class comp2_cupy_mesh(CuPyMultiComponentMesh):
    components = ['comp1', 'comp2']
