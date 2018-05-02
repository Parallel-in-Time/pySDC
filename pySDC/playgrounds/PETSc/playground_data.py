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
    y = 0.1*y + y
    print('y=', y.getArray())
    print((x-y).norm(PETSc.NormType.NORM_INFINITY))

    z = y.copy()
    print('z=', z.getArray())
    ya = da.getVecArray(y)
    ya[0,0] = 10.0
    print(y.getArray()[0], z.getArray()[0])

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
