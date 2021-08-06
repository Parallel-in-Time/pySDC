from petsc4py import PETSc


def main():

    n = 4
    da = PETSc.DMDA().create([n, n], stencil_width=1)

    rank = PETSc.COMM_WORLD.getRank()

    x = da.createGlobalVec()
    xa = da.getVecArray(x)
    (xs, xe), (ys, ye) = da.getRanges()
    print(da.getRanges())
    for i in range(xs, xe):
        for j in range(ys, ye):
            xa[i, j] = j*n + i
    print('x=', rank, x.getArray(), xs, xe, ys, ye)

    A = da.createMatrix()
    A.setType(PETSc.Mat.Type.FFTW)  # sparse
    A.setFromOptions()

    # Istart, Iend = A.getOwnershipRange()
    # for I in range(Istart, Iend):
    #     A[I, I] = 1.0

    # communicate off-processor values
    # and setup internal data structures
    # for performing parallel operations
    A.assemblyBegin()
    A.assemblyEnd()

    res = da.createGlobalVec()
    A.mult(x, res)
    print(rank, res.getArray())
    print((res-x).norm())


if __name__ == "__main__":
    main()
