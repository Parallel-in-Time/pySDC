# ------------------------------------------------------------------------
#
#  Poisson problem. This problem is modeled by the partial
#  differential equation
#
#          -Laplacian(u) = 1,  0 < x,y < 1,
#
#  with boundary conditions
#
#           u = 0  for  x = 0, x = 1, y = 0, y = 1
#
#  A finite difference approximation with the usual 7-point stencil
#  is used to discretize the boundary value problem to obtain a
#  nonlinear system of equations. The problem is solved in a 2D
#  rectangular domain, using distributed arrays (DAs) to partition
#  the parallel grid.
#
# ------------------------------------------------------------------------

# try: range = xrange
# except: pass

import numpy as np

import sys, petsc4py
petsc4py.init(sys.argv)

from petsc4py import PETSc

class Poisson2D(object):

    def __init__(self, da):
        assert da.getDim() == 2
        self.da = da
        self.localX  = da.createLocalVec()

    def formRHS(self, B, val):
        b = self.da.getVecArray(B)
        mx, my = self.da.getSizes()
        hx, hy = [1.0 / (m + 1) for m in [mx, my]]
        (xs, xe), (ys, ye) = self.da.getRanges()
        for j in range(ys, ye):
            for i in range(xs, xe):
                # b[i, j] = val*1*hx*hy
                b[i, j] = val * np.sin(2 * np.pi * (i + 1) * hx) * np.sin(2 * np.pi * (j + 1) * hy)

    def mult(self, mat, X, Y):
        #
        self.da.globalToLocal(X, self.localX)
        x = self.da.getVecArray(self.localX)
        y = self.da.getVecArray(Y)
        mx, my = self.da.getSizes()
        hx, hy = [1.0 / (m + 1) for m in [mx, my]]
        (xs, xe), (ys, ye) = self.da.getRanges()
        for j in range(ys, ye):
            for i in range(xs, xe):
                u = x[i, j] # center
                u_e = u_w = u_n = u_s = 0
                if i > 0:    u_w = x[i-1, j] # west
                if i < mx-1: u_e = x[i+1, j] # east
                if j > 0:    u_s = x[i, j-1] # south
                if j < my-1: u_n = x[i, j+1] # north
                u_xx = (u_e - 2*u + u_w) / hx ** 2
                u_yy = (u_n - 2*u + u_s) / hy ** 2
                y[i, j] = u_xx + u_yy

OptDB = PETSc.Options()

n  = OptDB.getInt('n', 4)
nx = OptDB.getInt('nx', n)
ny = OptDB.getInt('ny', n)

da = PETSc.DMDA().create([nx, ny], stencil_width=1)
pde = Poisson2D(da)

x = da.createGlobalVec()
b = da.createGlobalVec()
# A = da.createMat('python')
A = PETSc.Mat().createPython(
    [x.getSizes(), b.getSizes()], comm=da.comm)
print(A.getSize())
A.setPythonContext(pde)
A.setUp()

y = da.createGlobalVec()

pde.formRHS(x, val=1.0)
A.mult(x, b)
pde.formRHS(y, val=-2.0 * (2.0 * np.pi) ** 2)

# u = da.createNaturalVec()
# da.globalToNatural(b, u)

# print((b - y).norm(PETSc.NormType.NORM_INFINITY))
# exit()

ksp = PETSc.KSP().create()
ksp.setType('cg')
pc = ksp.getPC()
pc.setType('ilu')
ksp.setFromOptions()

ksp.setOperators(A)

x1 = da.createGlobalVec()
pde.formRHS(b,val=1)
ksp.solve(b, x1)

x2 = da.createGlobalVec()
pde.formRHS(b,val=0)
ksp.solve(b, x2)

print(x1.array)


u = da.createNaturalVec()
da.globalToNatural(x, u)

# if OptDB.getBool('plot', True):
#     draw = PETSc.Viewer.DRAW(x.comm)
#     OptDB['draw_pause'] = 1
#     draw(x)