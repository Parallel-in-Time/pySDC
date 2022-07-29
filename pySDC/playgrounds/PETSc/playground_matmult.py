import numpy as np
from petsc4py import PETSc

n = 4
dx = 1.0 / (n + 1)

da = PETSc.DMDA().create([n, n], stencil_width=1, comm=PETSc.COMM_WORLD)

# set up vectors
x = da.createNaturalVector()
b = x.duplicate()
y = x.duplicate()
xs, xe = x.getOwnershipRange()
ldim = x.getLocalSize()
for k in range(ldim):
    iglobal = k + xs
    j = k % n
    i = k // n
    # set up target
    x.setValue(iglobal, np.sin(2 * np.pi * (i + 1) * dx) * np.sin(2 * np.pi * (j + 1) * dx))
    # set up exact solution
    y.setValue(iglobal, -2 * (2.0 * np.pi) ** 2 * np.sin(2 * np.pi * (i + 1) * dx) * np.sin(2 * np.pi * (j + 1) * dx))
x.assemblyBegin()
x.assemblyEnd()

# set up 2nd order FD matrix, taken from https://bitbucket.org/petsc/petsc4py/src/master/demo/kspsolve/petsc-mat.py
A = da.createMatrix()
A.setType('aij')  # sparse
A.setFromOptions()
A.setPreallocationNNZ((5, 5))
A.setUp()


diagv = -2.0 / dx**2 - 2.0 / dx**2
offdx = 1.0 / dx**2
offdy = 1.0 / dx**2

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
A.assemblyBegin()
A.assemblyEnd()

A.mult(x, b)

rank = PETSc.COMM_WORLD.getRank()
print(rank, b.getArray())
print((b - y).norm(PETSc.NormType.NORM_INFINITY))
