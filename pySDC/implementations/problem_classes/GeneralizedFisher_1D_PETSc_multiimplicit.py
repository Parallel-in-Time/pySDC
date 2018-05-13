from __future__ import division
import numpy as np
from petsc4py import PETSc

from pySDC.core.Problem import ptype
from pySDC.core.Errors import ParameterError


class Fisher(object):
    """
    Helper class to generate residual and Jacobian matrix for PETSc's nonlinear solver SNES
    """
    def __init__(self, da, params, factor):
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
        self.params = params
        self.factor = factor
        self.localX = da.createLocalVec()

    def formFunction(self, snes, X, F):
        """
        Function to evaluate the residual for the Newton solver

        This function should be equal to the RHS in the solution

        Args:
            snes: nonlinear solver object
            X: input vector
            F: output vector F(X)

        Returns:
            None (overwrites F)
        """
        self.da.globalToLocal(X, self.localX)
        x = self.da.getVecArray(self.localX)
        f = self.da.getVecArray(F)
        mx = self.da.getSizes()[0]
        (xs, xe) = self.da.getRanges()[0]
        for i in range(xs, xe):
            if i == 0:
                f[i] = x[i]
            elif i == mx - 1:
                f[i] = x[i]
            else:
                f[i] = x[i] - self.factor * self.params.lambda0 ** 2 * x[i] * (1 - x[i] ** self.params.nu)

    def formJacobian(self, snes, X, J, P):
        """
        Function to return the Jacobian matrix

        Args:
            snes: nonlinear solver object
            X: input vector
            J: Jacobian matrix (current?)
            P: Jacobian matrix (new)

        Returns:
            matrix status
        """
        self.da.globalToLocal(X, self.localX)
        x = self.da.getVecArray(self.localX)
        P.zeroEntries()
        row = PETSc.Mat.Stencil()
        col = PETSc.Mat.Stencil()
        mx = self.da.getSizes()[0]
        (xs, xe) = self.da.getRanges()[0]
        for i in range(xs, xe):
            row.i = i
            row.field = 0
            if i == 0 or i == mx - 1:
                P.setValueStencil(row, row, 1.0)
            else:
                diag = 1.0 - self.factor * self.params.lambda0 ** 2 * (1.0 - (self.params.nu + 1) * x[i] ** self.params.nu)
                P.setValueStencil(row, row, diag)
        P.assemble()
        if J != P:
            J.assemble()  # matrix-free operator
        return PETSc.Mat.Structure.SAME_NONZERO_PATTERN


# noinspection PyUnusedLocal
class petsc_fisher(ptype):
    """
    Problem class implementing the fully implicit 2D Gray-Scott reaction-diffusion equation with periodic BC and PETSc
    """
    def __init__(self, problem_params, dtype_u, dtype_f):
        """
        Initialization routine

        Args:
            problem_params: custom parameters for the example
            dtype_u: particle data type (will be passed parent class)
            dtype_f: acceleration data type (will be passed parent class)
        """

        # define the Dirichlet boundary
        if 'comm' not in problem_params:
            problem_params['comm'] = PETSc.COMM_WORLD
        if 'sol_tol' not in problem_params:
            problem_params['sol_tol'] = 1E-10
        if 'sol_maxiter' not in problem_params:
            problem_params['sol_maxiter'] = None

        # these parameters will be used later, so assert their existence
        essential_keys = ['nvars', 'lambda0', 'nu', 'interval']
        for key in essential_keys:
            if key not in problem_params:
                msg = 'need %s to instantiate problem, only got %s' % (key, str(problem_params.keys()))
                raise ParameterError(msg)
        # create DMDA object which will be used for all grid operations (boundary_type=3 -> periodic BC)
        da = PETSc.DMDA().create([problem_params['nvars']], dof=1, stencil_width=1, comm=problem_params['comm'])

        # invoke super init, passing number of dofs, dtype_u and dtype_f
        super(petsc_fisher, self).__init__(init=da, dtype_u=dtype_u, dtype_f=dtype_f, params=problem_params)

        # compute dx, dy and get local ranges
        self.dx = (self.params.interval[1] - self.params.interval[0]) / (self.params.nvars - 1)
        # print(self.init.getRanges())
        (self.xs, self.xe) = self.init.getRanges()[0]

        # compute discretization matrix A and identity
        self.A = self.__get_A()
        self.localX = self.init.createLocalVec()

        # setup linear solver
        self.ksp = PETSc.KSP()
        self.ksp.create(comm=self.params.comm)
        self.ksp.setType('cg')
        pc = self.ksp.getPC()
        pc.setType('none')
        self.ksp.setInitialGuessNonzero(True)
        self.ksp.setFromOptions()
        self.ksp.setTolerances(rtol=self.params.sol_tol, atol=self.params.sol_tol, max_it=self.params.sol_maxiter)

        # setup nonlinear solver
        self.snes = PETSc.SNES()
        self.snes.create(comm=self.params.comm)
        # self.snes.getKSP().setType('cg')
        # self.snes.setType('ngmres')
        self.snes.setFromOptions()
        self.snes.setTolerances(rtol=self.params.sol_tol, atol=self.params.sol_tol, stol=self.params.sol_tol,
                                max_it=self.params.sol_maxiter)

    def __get_A(self):
        """
        Helper function to assemble PETSc matrix A

        Returns:
            PETSc matrix object
        """
        # create matrix and set basic options
        A = self.init.createMatrix()
        A.setType('aij')  # sparse
        A.setFromOptions()
        A.setPreallocationNNZ((3, 3))
        A.setUp()

        # fill matrix
        A.zeroEntries()
        row = PETSc.Mat.Stencil()
        col = PETSc.Mat.Stencil()
        mx = self.init.getSizes()[0]
        (xs, xe) = self.init.getRanges()[0]
        for i in range(xs, xe):
            row.i = i
            row.field = 0
            if i == 0 or i == mx - 1:
                A.setValueStencil(row, row, 1.0)
            else:
                diag = -2.0 / self.dx ** 2
                for index, value in [
                    (i - 1, 1.0 / self.dx ** 2),
                    (i, diag),
                    (i + 1, 1.0 / self.dx ** 2),
                ]:
                    col.i = index
                    col.field = 0
                    A.setValueStencil(row, col, value)
        A.assemble()
        return A

    def __get_sys_mat(self, factor):
        """
        Helper function to assemble PETSc identity matrix

        Returns:
            PETSc matrix object
        """

        # create matrix and set basic options
        A = self.init.createMatrix()
        A.setType('aij')  # sparse
        A.setFromOptions()
        A.setPreallocationNNZ((3, 3))
        A.setUp()

        # fill matrix
        A.zeroEntries()
        row = PETSc.Mat.Stencil()
        col = PETSc.Mat.Stencil()
        mx = self.init.getSizes()[0]
        (xs, xe) = self.init.getRanges()[0]
        for i in range(xs, xe):
            row.i = i
            row.field = 0
            if i == 0 or i == mx - 1:
                A.setValueStencil(row, row, 1.0)
            else:
                diag = 1.0 + factor * 2.0 / self.dx ** 2
                for index, value in [
                    (i - 1, -factor / self.dx ** 2),
                    (i, diag),
                    (i + 1, -factor / self.dx ** 2),
                ]:
                    col.i = index
                    col.field = 0
                    A.setValueStencil(row, col, value)
        A.assemble()
        return A

    def eval_f(self, u, t):
        """
        Routine to evaluate the RHS

        Args:
            u (dtype_u): current values
            t (float): current time

        Returns:
            dtype_f: the RHS
        """

        f = self.dtype_f(self.init)
        self.A.mult(u.values, f.comp1.values)
        fa1 = self.init.getVecArray(f.comp1.values)
        fa1[0] = 0
        fa1[-1] = 0

        fa2 = self.init.getVecArray(f.comp2.values)
        xa = self.init.getVecArray(u.values)
        for i in range(self.xs, self.xe):
            fa2[i] = self.params.lambda0 ** 2 * xa[i] * (1 - xa[i] ** self.params.nu)
        fa2[0] = 0
        fa2[-1] = 0
        # print('F:', fa[0], fa[-1])
        # exit()
        return f

    def solve_system_1(self, rhs, factor, u0, t):
        """
        Simple linear solver for (I-factor*A)u = rhs

        Args:
            rhs (dtype_f): right-hand side for the linear system
            factor (float): abbrev. for the local stepsize (or any other factor required)
            u0 (dtype_u): initial guess for the iterative solver
            t (float): current time (e.g. for time-dependent BCs)

        Returns:
            dtype_u: solution as mesh
        """

        me = self.dtype_u(u0)

        self.ksp.setOperators(self.__get_sys_mat(factor))
        self.ksp.solve(rhs.values, me.values)

        # xa = self.init.getVecArray(me.values)
        # print('x1', xa[0], xa[-1])

        return me

    def solve_system_2(self, rhs, factor, u0, t):
        """
        Nonlinear solver for (I-factor*F)(u) = rhs

        Args:
            rhs (dtype_f): right-hand side for the linear system
            factor (float): abbrev. for the local stepsize (or any other factor required)
            u0 (dtype_u): initial guess for the iterative solver
            t (float): current time (e.g. for time-dependent BCs)

        Returns:
            dtype_u: solution as mesh
        """

        me = self.dtype_u(u0)
        target = Fisher(self.init, self.params, factor)

        # ra = self.init.getVecArray(rhs.values)
        # print('r2', ra[0], ra[-1])
        # ra[0] = 0
        # ra[-1] = 1

        # assign residual function and Jacobian
        F = self.init.createGlobalVec()
        self.snes.setFunction(target.formFunction, F)
        J = self.init.createMatrix()
        self.snes.setJacobian(target.formJacobian, J)

        self.snes.solve(rhs.values, me.values)

        # xa = self.init.getVecArray(me.values)
        # print('x2', xa[0], xa[-1])

        print(self.snes.getConvergedReason(), self.snes.getLinearSolveIterations(), self.snes.getFunctionNorm(),
              self.snes.getKSP().getResidualNorm())
        return me

    def u_exact(self, t):
        """
        Routine to compute the exact solution at time t

        Args:
            t (float): current time

        Returns:
            dtype_u: exact solution
        """

        lam1 = self.params.lambda0 / 2.0 * ((self.params.nu / 2.0 + 1) ** 0.5 + (self.params.nu / 2.0 + 1) ** (-0.5))
        sig1 = lam1 - np.sqrt(lam1 ** 2 - self.params.lambda0 ** 2)
        me = self.dtype_u(self.init)
        xa = self.init.getVecArray(me.values)
        for i in range(self.xs, self.xe):
            xa[i] = (1 + (2 ** (self.params.nu / 2.0) - 1) * np.exp(-self.params.nu / 2.0 * sig1 * (self.params.interval[0] + (i + 1) * self.dx + 2 * lam1 * t))) ** (-2.0 / self.params.nu)
            # xa[i] = np.sin((self.params.interval[0] + i * self.dx)*np.pi/10)
            # print(xa[i])
        # print(xa[0], xa[-1])
        return me
