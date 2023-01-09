from dolfinx import fem, mesh
from mpi4py import MPI
import numpy as np


class fenics_mesh(fem.Function):
    """
    Numpy-based datatype for serial or parallel meshes.
    Can include a communicator and expects a dtype to allow complex data.

    Attributes:
        _comm: MPI communicator or None
    """

    def __new__(cls, init, val=0.0):
        """
        Instantiates new datatype. This ensures that even when manipulating data, the result is still a mesh.

        Args:
            init: either another mesh or a tuple containing the dimensions, the communicator and the dtype
            val: value to initialize

        Returns:
            obj of type mesh

        """
        if isinstance(init, fenics_mesh):
            obj = init.copy()
        elif (
            isinstance(init, fem.FunctionSpace)
            # isinstance(init, tuple)
            # and (init[1] is None or isinstance(init[1], MPI.Intracomm))
            # and isinstance(init[2], np.dtype)
        ):
            obj = fem.Function.__new__(cls)
            print(obj)
            # obj = fem.Function(init)
            obj.x.array[:] = val
        elif (
            isinstance(init, fem.function.Function)
        ):
            obj = init.copy()
        else:
            raise NotImplementedError(type(init))
        return obj

    def interpolate(self, u, cells=None):
        tmp = fem.Function(self.function_space)
        tmp.interpolate(u)
        return fenics_mesh(tmp)

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
        return comm.Issend(self.x.array[:], dest=dest, tag=tag)

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
        return comm.Irecv(self.x.array[:], source=source, tag=tag)

def initial_condition(x, a=5):
    return np.exp(-a*(x[0]**2+x[1]**2))


comm = MPI.COMM_WORLD
comm_fenics = comm.Split(color=comm.Get_rank())
print(comm_fenics.Get_size())
# Define mesh
nx, ny = 50, 50
domain = mesh.create_rectangle(comm_fenics, [np.array([-2, -2]), np.array([2, 2])],
                               [nx, ny], mesh.CellType.triangle)
V = fem.FunctionSpace(domain, ("CG", 1))

a = fenics_mesh(V)
# a.interpolate(initial_condition)
print(type(a))

b = fenics_mesh(a)
print(type(b))

d = a + b
print(type(d))

w_n = fem.Function(V)
w_n.name = "w_n"
w_n.x.array[:] = 99

print(np.mean(w_n.x.array))

if comm.Get_size() == 2:
    if comm.Get_rank() == 0:
        comm.send(u_n.x.array, dest=1)
    else:
        v_n.x.array[:] = comm.recv(source=0)
