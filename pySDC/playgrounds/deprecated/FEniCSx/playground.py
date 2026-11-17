from dolfinx import fem, mesh
from mpi4py import MPI
import numpy as np



def initial_condition(x, a=5):
    return np.exp(-a*(x[0]**2+x[1]**2))


comm = MPI.COMM_WORLD
comm_fenics = comm.Split(color=comm.Get_rank())
print(comm_fenics.Get_size())
# Define mesh
nx, ny = 50, 50
domain = mesh.create_rectangle(comm_fenics, [np.array([-2, -2]), np.array([2, 2])],
                               [nx, ny], mesh.CellType.triangle)
V = fem.FunctionSpace(domain, ("CG", 4))
nxc, nyc = 50, 50
domainc = mesh.create_rectangle(comm_fenics, [np.array([-2, -2]), np.array([2, 2])],
                               [nxc, nyc], mesh.CellType.triangle)
Vc = fem.FunctionSpace(domain, ("CG", 1))  ## WORKS
Vc = fem.FunctionSpace(domainc, ("CG", 1))  ## DOES NOT WORK

u = fem.Function(V)
uc = fem.Function(Vc)

u.interpolate(lambda x: initial_condition(x,a=5))

uc.interpolate(u)


# w_n = fem.Function(V)
# w_n.name = "w_n"
# w_n.x.array[:] = 99
#
# print(np.mean(w_n.x.array))
#
# if comm.Get_size() == 2:
#     if comm.Get_rank() == 0:
#         comm.send(u_n.x.array, dest=1)
#     else:
#         v_n.x.array[:] = comm.recv(source=0)
