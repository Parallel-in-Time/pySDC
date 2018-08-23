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

import pySDC.helpers.transfer_helper as th


def main():

    import petsc4py
    from petsc4py import PETSc

    n_fine = 5
    n_coarse = int((n_fine-1) / 2) + 1
    da_fine = PETSc.DMDA().create([n_fine, n_fine], stencil_width=1)
    da_coarse = PETSc.DMDA().create([n_coarse, n_coarse], stencil_width=1)

    x_fine = da_fine.createGlobalVec()
    xa = da_fine.getVecArray(x_fine)
    (xs, xe), (ys, ye) = da_fine.getRanges()
    nx, ny = da_fine.getSizes()
    for i in range(xs, xe):
        for j in range(ys, ye):
            # xa[i, j] = 1.0
            # xa[i, j] = i / nx
            xa[i, j] = np.sin(2 * np.pi * i / (nx+1)) * np.sin(2 * np.pi * j / (ny+1))

    da_coarse.setInterpolationType(PETSc.DMDA.InterpolationType.Q1)
    B, vec = da_coarse.createInterpolation(da_fine)


    # print(B, vec.getArray())

    x_coarse = da_coarse.createGlobalVec()
    xa = da_coarse.getVecArray(x_coarse)
    (xs, xe), (ys, ye) = da_coarse.getRanges()
    nx, ny = da_coarse.getSizes()
    for i in range(xs, xe):
        for j in range(ys, ye):
            xa[i, j] = 1.0
            # xa[i, j] = i / nx
            xa[i, j] = np.sin(2 * np.pi * i / (nx+1)) * np.sin(2 * np.pi * j / (ny+1))

    y = da_fine.createGlobalVec()
    # x_coarse.pointwiseMult(x_coarse)
    # PETSc.Mat.Restrict(B, x_coarse, y)
    B.mult(x_coarse, y)
    # y.pointwiseMult(vec, y)
    # PETSc.VecPointwiseMult()
    # print(y.getArray())
    # print(x_coarse.getArray())
    print((y-x_fine).norm(PETSc.NormType.NORM_INFINITY))

    y_coarse = da_coarse.createGlobalVec()
    B.multTranspose(x_fine, y_coarse)
    y_coarse.pointwiseMult(vec, y_coarse)

    print((y_coarse - x_coarse).norm(PETSc.NormType.NORM_INFINITY))


if __name__ == "__main__":
    main()
