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


class Poisson2D(object):

    def __init__(self, da):
        assert da.getDim() == 2
        self.da = da
        self.localX = da.createLocalVec()

    def formRHS(self, B):
        b = self.da.getVecArray(B)
        mx, my = self.da.getSizes()
        hx, hy = [1.0 / m for m in [mx, my]]
        (xs, xe), (ys, ye) = self.da.getRanges()
        for j in range(ys, ye):
            for i in range(xs, xe):
                b[i, j] = 1 * hx * hy

    def mult(self, mat, X, Y):
        #
        self.da.globalToLocal(X, self.localX)
        x = self.da.getVecArray(self.localX)
        y = self.da.getVecArray(Y)
        #
        mx, my = self.da.getSizes()
        hx, hy = [1.0 / m for m in [mx, my]]
        (xs, xe), (ys, ye) = self.da.getRanges()

        for j in range(ys, ye):
            for i in range(xs, xe):
                u = x[i, j]  # center
                u_e = u_w = u_n = u_s = 0
                if i > 0:    u_w = x[i - 1, j]  # west
                if i < mx - 1: u_e = x[i + 1, j]  # east
                if j > 0:    u_s = x[i, j - 1]  # south
                if j < my - 1: u_n = x[i, j + 1]  # north
                u_xx = (-u_e + 2 * u - u_w) * hy / hx
                u_yy = (-u_n + 2 * u - u_s) * hx / hy
                y[i, j] = u_xx + u_yy


def main():
    import petsc4py
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

    OptDB = PETSc.Options()

    n = OptDB.getInt('n', 16)
    nx = OptDB.getInt('nx', n)
    ny = OptDB.getInt('ny', n)

    t0 = time.time()
    da = PETSc.DMDA().create([nx, ny], stencil_width=1, comm=space_comm)
    pde = Poisson2D(da)

    x = da.createGlobalVec()
    b = da.createGlobalVec()
    # A = da.createMat('python')
    A = PETSc.Mat().createPython(
        [x.getSizes(), b.getSizes()], comm=space_comm)
    A.setPythonContext(pde)
    A.setUp()
    # print(da.comm, space_comm)

    ksp = PETSc.KSP().create(comm=space_comm)
    ksp.setOperators(A)
    ksp.setType('cg')
    pc = ksp.getPC()
    pc.setType('none')
    ksp.setFromOptions()

    t1 = time.time()

    pde.formRHS(b)
    ksp.solve(b, x)

    u = da.createNaturalVec()
    da.globalToNatural(x, u)

    t2 = time.time()

    n = 8

    rank = PETSc.COMM_WORLD.Get_rank()
    size = PETSc.COMM_WORLD.Get_size()

    print('Hello World! From process {rank} out of {size} process(es).'.format(rank=rank, size=size))

    x = PETSc.Vec().createSeq(n)  # Faster way to create a sequential vector.

    x.setValues(range(n), range(n))  # x = [0 1 ... 9]
    x.shift(1)  # x = x + 1 (add 1 to all elements in x)

    print('Performing various vector operations on x =', x.getArray())

    print('Sum of elements of x =', x.sum())

    print(t1-t0, t2-t1, t2-t0)

if __name__ == "__main__":
    main()
