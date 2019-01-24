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

import numpy as np
from mpi4py import MPI


def main():
    from petsc4py import PETSc

    # set MPI communicator
    comm = MPI.COMM_WORLD

    world_rank = comm.Get_rank()
    world_size = comm.Get_size()

    if len(sys.argv) == 2:
        color = int(world_rank / int(sys.argv[1]))
    else:
        color = int(world_rank / 1)

    space_comm = comm.Split(color=color)
    space_rank = space_comm.Get_rank()
    space_size = space_comm.Get_size()

    if len(sys.argv) == 2:
        color = int(world_rank % int(sys.argv[1]))
    else:
        color = int(world_rank / world_size)

    time_comm = comm.Split(color=color)
    time_rank = time_comm.Get_rank()
    time_size = time_comm.Get_size()

    print("IDs (world, space, time):  %i / %i -- %i / %i -- %i / %i" % (world_rank, world_size, space_rank, space_size,
                                                                        time_rank, time_size))

    n = 7
    da = PETSc.DMDA().create([n, n], stencil_width=1, comm=space_comm)

    x = da.createGlobalVec()
    xa = da.getVecArray(x)
    (xs, xe), (ys, ye) = da.getRanges()
    for i in range(xs, xe):
        for j in range(ys, ye):
            xa[i, j] = np.sin(2 * np.pi * (i + 1) / (n + 1)) * np.sin(2 * np.pi * (j + 1) / (n + 1))
    print('x=', x.getArray())
    print('x:', x.getSizes(), da.getRanges())
    print()

    if time_rank == 0:
        print('send', time_rank)
        time_comm.send(x.getArray(), dest=1, tag=0)
    else:
        print('recv', time_rank)
        y = da.createGlobalVec()
        y.setArray(time_comm.recv(source=0, tag=0))
        print(type(y.getArray()))


if __name__ == "__main__":
    main()
