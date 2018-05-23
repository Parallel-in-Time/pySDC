# Summary
#     Creating and using vectors and basic vector operations in PETSc.
#
# Description
#     Vectors are a basic mathematical building block.
#
# For a complete list of vector operations, consult the PETSc user manual.
# Also look at the petsc4py/src/PETSc/Vec.pyx file for petsc4py implementation
# details.

import sys

from mpi4py import MPI
import time
from dolfin import *



def main():
    # # set MPI communicator
    # comm = MPI.COMM_WORLD
    #
    # world_rank = comm.Get_rank()
    # world_size = comm.Get_size()
    #
    # if len(sys.argv) == 2:
    #     color = int(world_rank / int(sys.argv[1]))
    # else:
    #     color = int(world_rank / 1)
    #
    # space_comm = comm.Split(color=color)
    # space_rank = space_comm.Get_rank()
    # space_size = space_comm.Get_size()
    #
    # if len(sys.argv) == 2:
    #     color = int(world_rank % int(sys.argv[1]))
    # else:
    #     color = int(world_rank / world_size)
    #
    # time_comm = comm.Split(color=color)
    # time_rank = time_comm.Get_rank()
    # time_size = time_comm.Get_size()
    #
    # print("IDs (world, space, time):  %i / %i -- %i / %i -- %i / %i" % (world_rank, world_size, space_rank, space_size,
    #                                                                     time_rank, time_size))

    # Create mesh and define function space
    mesh = UnitSquareMesh(32, 32)
    V = FunctionSpace(mesh, "Lagrange", 1)

    # Define Dirichlet boundary (x = 0 or x = 1)
    def boundary(x):
        return x[0] < DOLFIN_EPS or x[0] > 1.0 - DOLFIN_EPS

    # Define boundary condition
    u0 = Constant(0.0)
    bc = DirichletBC(V, u0, boundary)

    # Define variational problem
    u = TrialFunction(V)
    v = TestFunction(V)
    f = Expression("10*exp(-(pow(x[0] - 0.5, 2) + pow(x[1] - 0.5, 2)) / 0.02)")
    g = Expression("sin(5*x[0])")
    a = inner(grad(u), grad(v)) * dx
    L = f * v * dx + g * v * ds

    # Compute solution
    u = Function(V)
    solve(a == L, u, bc)

    # print(t1-t0, t2-t1, t2-t0)

if __name__ == "__main__":
    main()
