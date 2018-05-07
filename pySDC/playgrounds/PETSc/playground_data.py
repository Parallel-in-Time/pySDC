
import numpy as np

from petsc4py import PETSc


def main():
    # import petsc4py


    n = 4
    dx = 1.0/(n - 1)
    dy = dx
    comm= PETSc.COMM_WORLD
    da = PETSc.DMDA().create([n, n], stencil_width=1, comm=comm)

    rank = PETSc.COMM_WORLD.getRank()
    # comm=

    x = da.createGlobalVector()
    # x.setSizes(n*n)
    y = x.duplicate()
    res = x.duplicate()
    xs, xe = x.getOwnershipRange()
    ldim = x.getLocalSize()
    print(xs, xe, ldim)
    for k in range(ldim):
        iglobal = k + xs
        j = k % n
        i = k // n
        x.setValue(iglobal, np.sin(2 * np.pi * (i ) * dx) * np.sin(2 * np.pi * (j ) * dy))
        y.setValue(iglobal, -2 * (2.0 * np.pi) ** 2 * np.sin(2 * np.pi * (i ) * dx) * np.sin(2 * np.pi * (j ) * dy))
    x.assemblyBegin()
    x.assemblyEnd()
    #
    x = da.createGlobalVec()
    xa = da.getVecArray(x)
    (xs, xe), (ys, ye) = da.getRanges()
    for i in range(xs, xe):
        for j in range(ys, ye):
            xa[i, j] = np.sin(2 * np.pi * (i ) * dx) * np.sin(2 * np.pi * (j ) * dy)
    print('x=', rank, x.getArray())
    # print('x:', x.getSizes(), da.getRanges())
    # print()

    y = da.createGlobalVec()
    ya = da.getVecArray(y)
    (xs, xe), (ys, ye) = da.getRanges()
    for i in range(xs, xe):
        for j in range(ys, ye):
            ya[i, j] = -2 * (2.0 * np.pi) ** 2 * np.sin(2 * np.pi * (i ) * dx) * np.sin(2 * np.pi * (j ) * dy)
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

    A = da.createMatrix()
    A.setType('aij')  # sparse
    A.setFromOptions()
    A.setPreallocationNNZ((5,5))
    A.setUp()

    A.zeroEntries()
    row = PETSc.Mat.Stencil()
    col = PETSc.Mat.Stencil()
    mx, my = da.getSizes()
    (xs, xe), (ys, ye) = da.getRanges()
    for j in range(ys, ye):
        for i in range(xs, xe):
            row.index = (i, j)
            row.field = 0
            if (i == 0 or j == 0 or i == mx - 1 or j == my - 1):
                A.setValueStencil(row, row, 1.0)
                # pass
            else:
                # u = x[i, j] # center
                diag = -2.0 / dx ** 2 - 2.0 / dy ** 2
                for index, value in [
                    ((i, j - 1), 1.0 / dy ** 2),
                    ((i - 1, j), 1.0 / dx ** 2),
                    ((i, j), diag),
                    ((i + 1, j), 1.0 / dx ** 2),
                    ((i, j + 1), 1.0 / dy ** 2),
                ]:
                    col.index = index
                    col.field = 0
                    A.setValueStencil(row, col, value)
    A.assemble()

    Id = da.createMatrix()
    Id.setType('aij')  # sparse
    Id.setFromOptions()
    Id.setPreallocationNNZ((5, 5))
    Id.setUp()

    Id.zeroEntries()
    row = PETSc.Mat.Stencil()
    col = PETSc.Mat.Stencil()
    mx, my = da.getSizes()
    (xs, xe), (ys, ye) = da.getRanges()
    for j in range(ys, ye):
        for i in range(xs, xe):
            row.index = (i, j)
            row.field = 0
            col.index = (i, j)
            col.field = 0
            Id.setValueStencil(row, row, 1.0)
            # if (i == 0 or j == 0 or i == mx - 1 or j == my - 1):
            #     Id.setValueStencil(row, row, 1.0)
            #     pass
            # else:
            #
            #     col.field = 0
            #     Id.setValueStencil(row, col, 1.0)
                # # u = x[i, j] # center
                # diag = 1.0
                # for index, value in [
                #     # ((i, j - 1), 1.0 / dy ** 2),
                #     # ((i - 1, j), 1.0 / dx ** 2),
                #     ((i, j), diag),
                #     # ((i + 1, j), 1.0 / dx ** 2),
                #     # ((i, j + 1), 1.0 / dy ** 2),
                # ]:
                #     col.index = index
                #     col.field = 0
                #     Id.setValueStencil(row, col, value)
    Id.assemble()

    # (xs, xe), (ys, ye) = da.getRanges()
    # print(A.getValues(range(n*n), range(n*n)))

    A.mult(x, res)
    print('1st turn', rank, res.getArray())
    print((res-y).norm(PETSc.NormType.NORM_INFINITY))

    ksp = PETSc.KSP().create()
    ksp.setOperators(A)
    ksp.setType('cg')
    pc = ksp.getPC()
    pc.setType('mg')
    ksp.setFromOptions()

    x1 = da.createGlobalVec()
    ksp.solve(res, x1)
    print((x1 - x).norm(PETSc.NormType.NORM_INFINITY))

    x2 = da.createGlobalVec()
    Id.mult(x1, x2)
    print((x2 - x1).norm(PETSc.NormType.NORM_INFINITY))


    # # A.view()
    # res1 = da.createNaturalVec()
    # A.mult(res, res1)
    # # print('2nd turn', rank, res1.getArray())
    # da.globalToNatural(res, res1)
    # print(res1.getArray())
    # print((res1 - y).norm(PETSc.NormType.NORM_INFINITY))


if __name__ == "__main__":
    main()
