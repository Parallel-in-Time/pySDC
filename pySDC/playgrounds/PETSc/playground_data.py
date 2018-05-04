
import numpy as np

from petsc4py import PETSc


def main():
    # import petsc4py


    n = 4
    dx = 1.0/(n + 1)
    dy = dx
    da = PETSc.DMDA().create([n, n], stencil_width=1)

    rank = PETSc.COMM_WORLD.getRank()
    comm=PETSc.COMM_WORLD

    x = PETSc.Vec().createMPI(n*n, comm=comm)
    y = x.duplicate()
    res = x.duplicate()
    xs, xe = x.getOwnershipRange()
    ldim = x.getLocalSize()
    for k in range(ldim):
        iglobal = k + xs
        j = k % n
        i = k // n
        x.setValue(iglobal, np.sin(2 * np.pi * (i + 1) * dx) * np.sin(2 * np.pi * (j + 1) * dy))
        y.setValue(iglobal, -2 * (2.0 * np.pi) ** 2 * np.sin(2 * np.pi * (i + 1) * dx) * np.sin(2 * np.pi * (j + 1) * dy))
    x.assemblyBegin()
    x.assemblyEnd()
    #
    # x = da.createGlobalVec()
    # xa = da.getVecArray(x)
    # (xs, xe), (ys, ye) = da.getRanges()
    # for i in range(xs, xe):
    #     for j in range(ys, ye):
    #         xa[i, j] = np.sin(2 * np.pi * (i + 1) * dx) * np.sin(2 * np.pi * (j + 1) * dy)
    print('x=', rank, x.getArray())
    # print('x:', x.getSizes(), da.getRanges())
    # print()

    # y = da.createGlobalVec()
    # ya = da.getVecArray(y)
    # (xs, xe), (ys, ye) = da.getRanges()
    # for i in range(xs, xe):
    #     for j in range(ys, ye):
    #         ya[i, j] =
    #
    # z = da.createGlobalVec()
    # za = da.getVecArray(z)
    # (xs, xe), (ys, ye) = da.getRanges()
    # for i in range(xs, xe):
    #     for j in range(ys, ye):
    #         za[i, j] = 4 * (2.0 * np.pi) ** 4 * np.sin(2 * np.pi * (i + 1) * dx) * np.sin(2 * np.pi * (j + 1) * dy)


    # z = y.copy()
    # print('z=', z.getArray())
    # ya = da.getVecArray(y)
    # ya[0,0] = 10.0
    # print(y.getArray()[0], z.getArray()[0])

    # A = da.createMatrix()
    # A.setType('aij')  # sparse
    # A.setFromOptions()
    # A.setPreallocationNNZ((5,5))
    # A.setUp()

    A = PETSc.Mat().create(comm=comm)
    A.setSizes(n*n)
    # A.setType('aij')  # sparse
    A.setFromOptions()
    # A.setPreallocationNNZ((5,5))
    A.setUp()

    diagv = -2.0 / dx ** 2 - 2.0 / dy ** 2
    offdx = (1.0 / dx ** 2)
    offdy = (1.0 / dy ** 2)

    Istart, Iend = A.getOwnershipRange()
    for I in range(Istart, Iend):
        A.setValue(I, I, diagv)
        i = I // n  # map row number to
        j = I - i * n  # grid coordinates
        if i > 0:
            J = I - n
            A.setValues(I, J, offdx)
        if i < n - 1:
            J = I + n
            A.setValues(I, J, offdx)
        if j > 0:
            J = I - 1
            A.setValues(I, J, offdy)
        if j < n - 1:
            J = I + 1
            A.setValues(I, J, offdy)

    # communicate off-processor values
    # and setup internal data structures
    # for performing parallel operations
    A.assemblyBegin()
    A.assemblyEnd()

    # (xs, xe), (ys, ye) = da.getRanges()
    # print(A.getValues(range(n*n), range(n*n)))

    A.mult(x, res)
    print('1st turn', rank, res.getArray())
    res.axpy(-1.0, y)
    print(res.norm(PETSc.NormType.NORM_INFINITY))

    # # A.view()
    # res1 = da.createGlobalVec()
    # A.mult(res, res1)
    # # print('2nd turn', rank, res1.getArray())
    # # da.globalToNatural(res1, u1)
    # print((res1 - z).norm(PETSc.NormType.NORM_INFINITY))


if __name__ == "__main__":
    main()
