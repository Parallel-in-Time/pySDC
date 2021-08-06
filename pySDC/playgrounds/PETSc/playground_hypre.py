import petsc4py, sys
# args = "-ksp_type cg -pc_type hypre -pc_hypre_type boomeramg -ksp_converged_reason"
# args = "-ksp_type gmres -pc_type gamg -ksp_converged_reason -ksp_monitor"
args = "-ksp_type richardson -pc_type gamg -ksp_converged_reason -ksp_monitor"
petsc4py.init(args)

from petsc4py import PETSc


# grid size and spacing
for grid_sp in [500]:
    m, n  = grid_sp, grid_sp
    hx = 1.0/(m-1)
    hy = 1.0/(n-1)

    # create sparse matrix
    A = PETSc.Mat()
    A.create(PETSc.COMM_WORLD)
    A.setSizes([m*n, m*n])
    A.setType('aij') # sparse
    A.setPreallocationNNZ(5)

    # precompute values for setting
    # diagonal and non-diagonal entries
    diagv = 2.0/hx**2 + 2.0/hy**2
    offdx = -1.0/hx**2
    offdy = -1.0/hy**2

    # loop over owned block of rows on this
    # processor and insert entry values
    Istart, Iend = A.getOwnershipRange()
    print(Istart, Iend)
    for I in range(Istart, Iend) :
        A[I,I] = diagv
        i = I//n    # map row number to
        j = I - i*n # grid coordinates
        if i> 0  : J = I-n; A[I,J] = offdx
        if i< m-1: J = I+n; A[I,J] = offdx
        if j> 0  : J = I-1; A[I,J] = offdy
        if j< n-1: J = I+1; A[I,J] = offdy

    # communicate off-processor values
    # and setup internal data structures
    # for performing parallel operations
    A.assemblyBegin()
    A.assemblyEnd()
    # print(A.isSymmetric())

    # create linear solver
    ksp = PETSc.KSP()
    ksp.create(PETSc.COMM_WORLD)
    # obtain sol & rhs vectors
    x, b = A.createVecs()
    print(type(x))
    exit()
    x.set(0)
    b.set(1)
    # and next solve
    ksp.setOperators(A)
    ksp.setFromOptions()
    ksp.solve(b, x)
    # x.set(0)
    # ksp.solveTranspose(b, x)
