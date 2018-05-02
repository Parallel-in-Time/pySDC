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
import numpy as np
from matplotlib import pylab


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

    n = 7
    da = PETSc.DMDA().create([n, n], stencil_width=1)

    x = da.createGlobalVec()
    xa = da.getVecArray(x)
    (xs, xe), (ys, ye) = da.getRanges()
    for i in range(xs, xe):
        for j in range(ys, ye):
            xa[i, j] = np.sin(2 * np.pi * (i + 1) / (n + 1)) * np.sin(2 * np.pi * (j + 1) / (n + 1))
    print('x=', x.getArray())
    print('x:', x.getSizes(), da.getRanges())
    print()

    y = da.createGlobalVec()
    ya = da.getVecArray(y)
    (xs, xe), (ys, ye) = da.getRanges()
    for i in range(xs, xe):
        for j in range(ys, ye):
            ya[i, j] = np.sin(2 * np.pi * (i + 1) / (n + 1)) * np.sin(2 * np.pi * (j + 1) / (n + 1))
    y = 0.1*y
    print('y=', y.getArray())
    print((x-y).norm(PETSc.NormType.NORM_INFINITY))

    # y = da.createLocalVec()
    # da.globalToLocal(x, y)
    # print('y=', y.getArray())
    # print('y:', y.getSizes())

    # pylab.imshow(x.getArray().reshape((ye-ys,xe-xs)))
    # pylab.show()

    # t2 = time.time()
    #
    # n = 8
    #
    # rank = PETSc.COMM_WORLD.Get_rank()
    # size = PETSc.COMM_WORLD.Get_size()
    #
    # print('Hello World! From process {rank} out of {size} process(es).'.format(rank=rank, size=size))
    #
    # x = PETSc.Vec().createSeq(n)  # Faster way to create a sequential vector.
    #
    # x.setValues(range(n), range(n))  # x = [0 1 ... 9]
    # x.shift(1)  # x = x + 1 (add 1 to all elements in x)
    #
    # print('Performing various vector operations on x =', x.getArray())
    #
    # print('Sum of elements of x =', x.sum())
    #
    # print(t1-t0, t2-t1, t2-t0)

if __name__ == "__main__":
    main()
