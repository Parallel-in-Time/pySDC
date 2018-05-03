
import numpy as np
from petsc4py import PETSc


def main():
    # import petsc4py


    n = 4
    dx = 1.0/(n + 1)
    dy = dx
    da = PETSc.DMDA().create([n, n], stencil_width=2)

    rank = PETSc.COMM_WORLD.getRank()

    x = da.createGlobalVec()
    xa = da.getVecArray(x)
    (xs, xe), (ys, ye) = da.getRanges()
    for i in range(xs, xe):
        for j in range(ys, ye):
            xa[i, j] = np.sin(2 * np.pi * (i + 1) * dx) * np.sin(2 * np.pi * (j + 1) * dy)
    print('x=', rank, x.getArray())
    # print('x:', x.getSizes(), da.getRanges())
    # print()

    y = da.createGlobalVec()
    ya = da.getVecArray(y)
    (xs, xe), (ys, ye) = da.getRanges()
    for i in range(xs, xe):
        for j in range(ys, ye):
            ya[i, j] = -2 * (2.0 * np.pi) ** 2 * np.sin(2 * np.pi * (i + 1) * dx) * np.sin(2 * np.pi * (j + 1) * dy)

    z = da.createGlobalVec()
    za = da.getVecArray(z)
    (xs, xe), (ys, ye) = da.getRanges()
    for i in range(xs, xe):
        for j in range(ys, ye):
            za[i, j] = 4 * (2.0 * np.pi) ** 4 * np.sin(2 * np.pi * (i + 1) * dx) * np.sin(2 * np.pi * (j + 1) * dy)


    # z = y.copy()
    # print('z=', z.getArray())
    # ya = da.getVecArray(y)
    # ya[0,0] = 10.0
    # print(y.getArray()[0], z.getArray()[0])

    A = da.createMatrix()
    A.setType('aij')  # sparse
    A.setFromOptions()
    A.setPreallocationNNZ((5,5))

    diagv = -2.0 / dx ** 2 - 2.0 / dy ** 2
    offdx = (1.0 / dx ** 2)
    offdy = (1.0 / dy ** 2)

    Istart, Iend = A.getOwnershipRange()
    for I in range(Istart, Iend):
        A[I, I] = diagv
        i = I // n  # map row number to
        j = I - i * n  # grid coordinates
        if i > 0:
            J = I - n
            A[I, J] = offdx
        if i < n - 1:
            J = I + n
            A[I, J] = offdx
        if j > 0:
            J = I - 1
            A[I, J] = offdy
        if j < n - 1:
            J = I + 1
            A[I, J] = offdy

    # communicate off-processor values
    # and setup internal data structures
    # for performing parallel operations
    A.assemblyBegin()
    A.assemblyEnd()

    # (xs, xe), (ys, ye) = da.getRanges()
    # print(A.getValues(range(n*n), range(n*n)))

    res = da.createGlobalVec()
    u = da.createNaturalVec()
    da.globalToNatural(x, u)
    A.mult(u, res)
    print('1st turn', rank, res.getArray())
    da.globalToNatural(res, u)
    print((u - y).norm(PETSc.NormType.NORM_INFINITY))

    res1 = da.createGlobalVec()
    u1 = da.createNaturalVec()
    da.globalToNatural(u, u1)
    A.mult(u1, res1)
    print('2nd turn', rank, res1.getArray())
    da.globalToNatural(res1, u1)
    print((u1 - z).norm(PETSc.NormType.NORM_INFINITY))


if __name__ == "__main__":
    main()
