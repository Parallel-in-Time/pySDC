# Solves Heat equation on a periodic domain, using raw VecScatter
import petsc4py
import sys
import time

petsc4py.init(sys.argv)

from petsc4py import PETSc
from mpi4py import MPI
import numpy as np

class Fisher_full(object):
    """
    Helper class to generate residual and Jacobian matrix for PETSc's nonlinear solver SNES
    """
    def __init__(self, da):
        """
        Initialization routine

        Args:
            da: DMDA object
            params: problem parameters
            factor: temporal factor (dt*Qd)
            dx: grid spacing in x direction
        """
        assert da.getDim() == 1
        self.da = da
        self.mx = self.da.getSizes()[0]
        (self.xs, self.xe) = self.da.getRanges()[0]
        self.dx = 100 / (self.mx - 1)
        print(self.mx, self.dx, self.xs, self.xe)

        self.lambda0 = 2.0
        self.nu = 1

        self.localX = self.da.createLocalVec()
        self.row = PETSc.Mat.Stencil()
        self.col = PETSc.Mat.Stencil()

        self.mat = self.da.createMatrix()
        self.mat.setType('aij')  # sparse
        self.mat.setFromOptions()
        self.mat.setPreallocationNNZ((3, 3))
        self.mat.setUp()

        self.gvec = self.da.createGlobalVec()

    def formFunction(self, ts, t, xin, xdot, f):
        self.da.globalToLocal(xin, self.localX)
        x = self.da.getVecArray(self.localX)
        fa = self.da.getVecArray(f)
        for i in range(self.xs, self.xe):
            if i == 0:
                fa[i] = x[i] - 0
            elif i == self.mx - 1:
                fa[i] = x[i] - 1
            else:
                u = x[i]  # center
                u_e = x[i + 1]  # east
                u_w = x[i - 1]  # west
                u_xx = (u_e - 2 * u + u_w) / self.dx ** 2
                fa[i] = xdot[i] - (u_xx + self.lambda0 ** 2 * x[i] * (1 - x[i] ** self.nu))

    def formJacobian(self, ts, t, xin, xdot, a, A, B):
        self.da.globalToLocal(xin, self.localX)
        x = self.da.getVecArray(self.localX)
        B.zeroEntries()

        for i in range(self.xs, self.xe):
            self.row.i = i
            self.row.field = 0
            if i == 0 or i == self.mx - 1:
                B.setValueStencil(self.row, self.row, 1.0)
            else:
                diag = a - (-2.0 / self.dx ** 2 + self.lambda0 ** 2 * (1.0 - (self.nu + 1) * x[i] ** self.nu))
                for index, value in [
                    (i - 1, -1.0 / self.dx ** 2),
                    (i, diag),
                    (i + 1, -1.0 / self.dx ** 2),
                ]:
                    self.col.i = index
                    self.col.field = 0
                    B.setValueStencil(self.row, self.col, value)
        B.assemble()
        if A != B:
            A.assemble()  # matrix-free operator
        return PETSc.Mat.Structure.SAME_NONZERO_PATTERN

    def evalSolution(self, t, x):
        lam1 = self.lambda0 / 2.0 * ((self.nu / 2.0 + 1) ** 0.5 + (self.nu / 2.0 + 1) ** (-0.5))
        sig1 = lam1 - np.sqrt(lam1 ** 2 - self.lambda0 ** 2)
        xa = self.da.getVecArray(x)
        for i in range(self.xs, self.xe):
            xa[i] = (1 + (2 ** (self.nu / 2.0) - 1) * np.exp(-self.nu / 2.0 * sig1 * (-50 + (i + 1) * self.dx + 2 * lam1 * t))) ** (-2.0 / self.nu)

class Fisher_split(object):
    """
    Helper class to generate residual and Jacobian matrix for PETSc's nonlinear solver SNES
    """
    def __init__(self, da):
        """
        Initialization routine

        Args:
            da: DMDA object
            params: problem parameters
            factor: temporal factor (dt*Qd)
            dx: grid spacing in x direction
        """
        assert da.getDim() == 1
        self.da = da
        self.mx = self.da.getSizes()[0]
        (self.xs, self.xe) = self.da.getRanges()[0]
        self.dx = 100 / (self.mx - 1)
        print(self.mx, self.dx, self.xs, self.xe)

        self.lambda0 = 2.0
        self.nu = 1

        self.localX = self.da.createLocalVec()
        self.row = PETSc.Mat.Stencil()
        self.col = PETSc.Mat.Stencil()

        self.mat = self.da.createMatrix()
        self.mat.setType('aij')  # sparse
        self.mat.setFromOptions()
        self.mat.setPreallocationNNZ((3, 3))
        self.mat.setUp()

        self.gvec = self.da.createGlobalVec()
        self.rhs = self.da.createGlobalVec()

    def formFunction(self, ts, t, xin, xdot, f):
        self.da.globalToLocal(xin, self.localX)
        x = self.da.getVecArray(self.localX)
        fa = self.da.getVecArray(f)
        for i in range(self.xs, self.xe):
            if i == 0:
                fa[i] = x[i] - 0
            elif i == self.mx - 1:
                fa[i] = x[i] - 1
            else:
                u = x[i]  # center
                u_e = x[i + 1]  # east
                u_w = x[i - 1]  # west
                u_xx = (u_e - 2 * u + u_w) / self.dx ** 2
                # fa[i] = xdot[i] - (u_xx + self.lambda0 ** 2 * x[i] * (1 - x[i] ** self.nu))
                fa[i] = xdot[i] - u_xx

    def formJacobian(self, ts, t, xin, xdot, a, A, B):
        self.da.globalToLocal(xin, self.localX)
        x = self.da.getVecArray(self.localX)
        B.zeroEntries()

        for i in range(self.xs, self.xe):
            self.row.i = i
            self.row.field = 0
            if i == 0 or i == self.mx - 1:
                B.setValueStencil(self.row, self.row, 1.0)
            else:
                # diag = a - (-2.0 / self.dx ** 2 + self.lambda0 ** 2 * (1.0 - (self.nu + 1) * x[i] ** self.nu))
                diag = a - (-2.0 / self.dx ** 2)
                for index, value in [
                    (i - 1, -1.0 / self.dx ** 2),
                    (i, diag),
                    (i + 1, -1.0 / self.dx ** 2),
                ]:
                    self.col.i = index
                    self.col.field = 0
                    B.setValueStencil(self.row, self.col, value)
        B.assemble()
        if A != B:
            A.assemble()  # matrix-free operator
        return PETSc.Mat.Structure.SAME_NONZERO_PATTERN

    # def formRHS(self, ts, t, x, F):
    #     # print ('MyODE.rhsfunction()')
    #     f = self.lambda0 ** 2 * x[:] * (1 - x[:] ** self.nu)
    #     f.copy(F)

    def formRHS(self, ts, t, xin, F):
        self.da.globalToLocal(xin, self.localX)
        x = self.da.getVecArray(self.localX)
        fa = self.da.getVecArray(F)
        for i in range(self.xs, self.xe):
            if i == 0:
                fa[i] = 0
            elif i == self.mx - 1:
                fa[i] = 1
            else:
                # fa[i] = xdot[i] - (u_xx + self.lambda0 ** 2 * x[i] * (1 - x[i] ** self.nu))
                fa[i] = self.lambda0 ** 2 * x[i] * (1 - x[i] ** self.nu)

    def evalSolution(self, t, x):
        lam1 = self.lambda0 / 2.0 * ((self.nu / 2.0 + 1) ** 0.5 + (self.nu / 2.0 + 1) ** (-0.5))
        sig1 = lam1 - np.sqrt(lam1 ** 2 - self.lambda0 ** 2)
        xa = self.da.getVecArray(x)
        for i in range(self.xs, self.xe):
            xa[i] = (1 + (2 ** (self.nu / 2.0) - 1) * np.exp(-self.nu / 2.0 * sig1 * (-50 + (i + 1) * self.dx + 2 * lam1 * t))) ** (-2.0 / self.nu)



da = PETSc.DMDA().create([2049], dof=1, stencil_width=1)

OptDB = PETSc.Options()
ode = Fisher_split(da=da)

x = ode.gvec.duplicate()
f = ode.gvec.duplicate()

ts = PETSc.TS().create(comm=MPI.COMM_WORLD)
ts.setType(ts.Type.ARKIMEXARS443)        # Rosenbrock-W. ARKIMEX is a nonlinearly implicit alternative.
# ts.setRKType('3bs')

ts.setIFunction(ode.formFunction, ode.gvec)
ts.setIJacobian(ode.formJacobian, ode.mat)
ts.setRHSFunction(ode.formRHS, ode.rhs)

# ts.setMonitor(ode.monitor)

ts.setTime(0.0)
ts.setTimeStep(0.25)
ts.setMaxTime(1.0)
ts.setMaxSteps(100)
ts.setExactFinalTime(PETSc.TS.ExactFinalTime.INTERPOLATE)
ts.setMaxSNESFailures(-1)       # allow an unlimited number of failures (step will be rejected and retried)
ts.setMaxStepRejections(-1)
ts.setTolerances(atol=1E-08)
snes = ts.getSNES()             # Nonlinear solver
snes.setTolerances(max_it=100)   # Stop nonlinear solve after 10 iterations (TS will retry with shorter step)
ksp = snes.getKSP()             # Linear solver
ksp.setType(ksp.Type.CG)        # Conjugate gradients
pc = ksp.getPC()                # Preconditioner
if True:                       # Configure algebraic multigrid, could use run-time options instead
    pc.setType(pc.Type.ILU)    # PETSc's native AMG implementation, mostly based on smoothed aggregation
    # OptDB['mg_coarse_pc_type'] = 'svd' # more specific multigrid options
    # OptDB['mg_levels_pc_type'] = 'sor'

ts.setFromOptions()             # Apply run-time options, e.g. -ts_adapt_monitor -ts_type arkimex -snes_converged_reason
ode.evalSolution(0.0, x)
t0 = time.perf_counter()

# pr = cProfile.Profile()
# pr.enable()
ts.solve(x)
# pr.disable()
# s = io.StringIO()
# sortby = 'cumulative'
# ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
# ps.print_stats()
# print(s.getvalue())
print('Time:', time.perf_counter() - t0)

uex = ode.gvec.duplicate()
ode.evalSolution(1.0, uex)
# uex.view()
# x.view()
print((uex-x).norm(PETSc.NormType.NORM_INFINITY))

print('steps %d (%d rejected, %d SNES fails), nonlinear its %d, linear its %d'
      % (ts.getStepNumber(), ts.getStepRejections(), ts.getSNESFailures(),
         ts.getSNESIterations(), ts.getKSPIterations()))

# if OptDB.getBool('plot_history', True) and ode.comm.rank == 0:
#     ode.plotHistory()