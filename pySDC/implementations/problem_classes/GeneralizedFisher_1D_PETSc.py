import numpy as np
from petsc4py import PETSc

from pySDC.core.Errors import ParameterError
from pySDC.core.Problem import ptype
from pySDC.implementations.datatype_classes.petsc_vec import petsc_vec, petsc_vec_imex, petsc_vec_comp2


class Fisher_full(object):
    """
    Helper class to generate residual and Jacobian matrix for PETSc's nonlinear solver SNES
    """

    def __init__(self, da, params, factor, dx):
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
        self.dx = dx
        self.localX = da.createLocalVec()
        (self.xs, self.xe) = self.da.getRanges()[0]
        self.mx = self.da.getSizes()[0]
        self.row = PETSc.Mat.Stencil()
        self.col = PETSc.Mat.Stencil()

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

        for i in range(self.xs, self.xe):
            if i == 0 or i == self.mx - 1:
                f[i] = x[i]
            else:
                u = x[i]  # center
                u_e = x[i + 1]  # east
                u_w = x[i - 1]  # west
                u_xx = (u_e - 2 * u + u_w) / self.dx**2
                f[i] = x[i] - self.factor * (u_xx + self.params.lambda0**2 * x[i] * (1 - x[i] ** self.params.nu))

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

        for i in range(self.xs, self.xe):
            self.row.i = i
            self.row.field = 0
            if i == 0 or i == self.mx - 1:
                P.setValueStencil(self.row, self.row, 1.0)
            else:
                diag = 1.0 - self.factor * (
                    -2.0 / self.dx**2
                    + self.params.lambda0**2 * (1.0 - (self.params.nu + 1) * x[i] ** self.params.nu)
                )
                for index, value in [
                    (i - 1, -self.factor / self.dx**2),
                    (i, diag),
                    (i + 1, -self.factor / self.dx**2),
                ]:
                    self.col.i = index
                    self.col.field = 0
                    P.setValueStencil(self.row, self.col, value)
        P.assemble()
        if J != P:
            J.assemble()  # matrix-free operator
        return PETSc.Mat.Structure.SAME_NONZERO_PATTERN


class Fisher_reaction(object):
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
            if i == 0 or i == mx - 1:
                f[i] = x[i]
            else:
                f[i] = x[i] - self.factor * self.params.lambda0**2 * x[i] * (1 - x[i] ** self.params.nu)

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
        mx = self.da.getSizes()[0]
        (xs, xe) = self.da.getRanges()[0]
        for i in range(xs, xe):
            row.i = i
            row.field = 0
            if i == 0 or i == mx - 1:
                P.setValueStencil(row, row, 1.0)
            else:
                diag = 1.0 - self.factor * self.params.lambda0**2 * (
                    1.0 - (self.params.nu + 1) * x[i] ** self.params.nu
                )
                P.setValueStencil(row, row, diag)
        P.assemble()
        if J != P:
            J.assemble()  # matrix-free operator
        return PETSc.Mat.Structure.SAME_NONZERO_PATTERN


class petsc_fisher_multiimplicit(ptype):
    """
    Problem class implementing the multi-implicit 1D generalized Fisher equation with periodic BC and PETSc
    """

    def __init__(self, problem_params, dtype_u=petsc_vec, dtype_f=petsc_vec_comp2):
        """
        Initialization routine

        Args:
            problem_params: custom parameters for the example
            dtype_u: PETSc data type (will be passed to parent class)
            dtype_f: PETSc data type with 2 components (will be passed to parent class)
        """

        # define optional parameters
        if 'comm' not in problem_params:
            problem_params['comm'] = PETSc.COMM_WORLD
        if 'lsol_tol' not in problem_params:
            problem_params['lsol_tol'] = 1e-10
        if 'nlsol_tol' not in problem_params:
            problem_params['nlsol_tol'] = 1e-10
        if 'lsol_maxiter' not in problem_params:
            problem_params['lsol_maxiter'] = None
        if 'nlsol_maxiter' not in problem_params:
            problem_params['nlsol_maxiter'] = None

        # these parameters will be used later, so assert their existence
        essential_keys = ['nvars', 'lambda0', 'nu', 'interval']
        for key in essential_keys:
            if key not in problem_params:
                msg = 'need %s to instantiate problem, only got %s' % (key, str(problem_params.keys()))
                raise ParameterError(msg)
        # create DMDA object which will be used for all grid operations
        da = PETSc.DMDA().create([problem_params['nvars']], dof=1, stencil_width=1, comm=problem_params['comm'])

        # invoke super init, passing number of dofs, dtype_u and dtype_f
        super(petsc_fisher_multiimplicit, self).__init__(
            init=da, dtype_u=dtype_u, dtype_f=dtype_f, params=problem_params
        )

        # compute dx and get local ranges
        self.dx = (self.params.interval[1] - self.params.interval[0]) / (self.params.nvars - 1)
        (self.xs, self.xe) = self.init.getRanges()[0]

        # compute discretization matrix A and identity
        self.A = self.__get_A()
        self.localX = self.init.createLocalVec()

        # setup linear solver
        self.ksp = PETSc.KSP()
        self.ksp.create(comm=self.params.comm)
        self.ksp.setType('cg')
        pc = self.ksp.getPC()
        pc.setType('ilu')
        self.ksp.setInitialGuessNonzero(True)
        self.ksp.setFromOptions()
        self.ksp.setTolerances(rtol=self.params.lsol_tol, atol=self.params.lsol_tol, max_it=self.params.lsol_maxiter)
        self.ksp_itercount = 0
        self.ksp_ncalls = 0

        # setup nonlinear solver
        self.snes = PETSc.SNES()
        self.snes.create(comm=self.params.comm)
        if self.params.nlsol_maxiter <= 1:
            self.snes.setType('ksponly')
        self.snes.getKSP().setType('cg')
        pc = self.snes.getKSP().getPC()
        pc.setType('ilu')
        # self.snes.setType('ngmres')
        self.snes.setFromOptions()
        self.snes.setTolerances(
            rtol=self.params.nlsol_tol,
            atol=self.params.nlsol_tol,
            stol=self.params.nlsol_tol,
            max_it=self.params.nlsol_maxiter,
        )
        self.snes_itercount = 0
        self.snes_ncalls = 0
        self.F = self.init.createGlobalVec()
        self.J = self.init.createMatrix()

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
                diag = -2.0 / self.dx**2
                for index, value in [
                    (i - 1, 1.0 / self.dx**2),
                    (i, diag),
                    (i + 1, 1.0 / self.dx**2),
                ]:
                    col.i = index
                    col.field = 0
                    A.setValueStencil(row, col, value)
        A.assemble()
        return A

    def get_sys_mat(self, factor):
        """
        Helper function to assemble the system matrix of the linear problem

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
                diag = 1.0 + factor * 2.0 / self.dx**2
                for index, value in [
                    (i - 1, -factor / self.dx**2),
                    (i, diag),
                    (i + 1, -factor / self.dx**2),
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
        self.A.mult(u, f.comp1)
        fa1 = self.init.getVecArray(f.comp1)
        fa1[0] = 0
        fa1[-1] = 0

        fa2 = self.init.getVecArray(f.comp2)
        xa = self.init.getVecArray(u)
        for i in range(self.xs, self.xe):
            fa2[i] = self.params.lambda0**2 * xa[i] * (1 - xa[i] ** self.params.nu)
        fa2[0] = 0
        fa2[-1] = 0

        return f

    def solve_system_1(self, rhs, factor, u0, t):
        """
        Linear solver for (I-factor*A)u = rhs

        Args:
            rhs (dtype_f): right-hand side for the linear system
            factor (float): abbrev. for the local stepsize (or any other factor required)
            u0 (dtype_u): initial guess for the iterative solver
            t (float): current time (e.g. for time-dependent BCs)

        Returns:
            dtype_u: solution
        """

        me = self.dtype_u(u0)

        self.ksp.setOperators(self.get_sys_mat(factor))
        self.ksp.solve(rhs, me)

        self.ksp_itercount += self.ksp.getIterationNumber()
        self.ksp_ncalls += 1

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
            dtype_u: solution
        """

        me = self.dtype_u(u0)
        target = Fisher_reaction(self.init, self.params, factor)

        # assign residual function and Jacobian
        F = self.init.createGlobalVec()
        self.snes.setFunction(target.formFunction, F)
        J = self.init.createMatrix()
        self.snes.setJacobian(target.formJacobian, J)

        self.snes.solve(rhs, me)

        self.snes_itercount += self.snes.getIterationNumber()
        self.snes_ncalls += 1

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
        sig1 = lam1 - np.sqrt(lam1**2 - self.params.lambda0**2)
        me = self.dtype_u(self.init)
        xa = self.init.getVecArray(me)
        for i in range(self.xs, self.xe):
            xa[i] = (
                1
                + (2 ** (self.params.nu / 2.0) - 1)
                * np.exp(-self.params.nu / 2.0 * sig1 * (self.params.interval[0] + (i + 1) * self.dx + 2 * lam1 * t))
            ) ** (-2.0 / self.params.nu)

        return me


class petsc_fisher_fullyimplicit(petsc_fisher_multiimplicit):
    """
    Problem class implementing the fully-implicit 2D Gray-Scott reaction-diffusion equation with periodic BC and PETSc
    """

    def __init__(self, problem_params, dtype_u=petsc_vec, dtype_f=petsc_vec):
        """
        Initialization routine

        Args:
            problem_params: custom parameters for the example
            dtype_u: PETSc data type (will be passed to parent class)
            dtype_f: PETSc data type (will be passed to parent class)
        """

        # invoke super init, passing number of dofs, dtype_u and dtype_f
        super(petsc_fisher_fullyimplicit, self).__init__(
            problem_params=problem_params, dtype_u=dtype_u, dtype_f=dtype_f
        )

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
        self.A.mult(u, f)

        fa2 = self.init.getVecArray(f)
        xa = self.init.getVecArray(u)
        for i in range(self.xs, self.xe):
            fa2[i] += self.params.lambda0**2 * xa[i] * (1 - xa[i] ** self.params.nu)
        fa2[0] = 0
        fa2[-1] = 0

        return f

    def solve_system(self, rhs, factor, u0, t):
        """
        Nonlinear solver for (I-factor*F)(u) = rhs

        Args:
            rhs (dtype_f): right-hand side for the linear system
            factor (float): abbrev. for the local stepsize (or any other factor required)
            u0 (dtype_u): initial guess for the iterative solver
            t (float): current time (e.g. for time-dependent BCs)

        Returns:
            dtype_u: solution
        """

        me = self.dtype_u(u0)
        target = Fisher_full(self.init, self.params, factor, self.dx)

        # assign residual function and Jacobian

        self.snes.setFunction(target.formFunction, self.F)
        self.snes.setJacobian(target.formJacobian, self.J)

        self.snes.solve(rhs, me)

        self.snes_itercount += self.snes.getIterationNumber()
        self.snes_ncalls += 1

        return me


class petsc_fisher_semiimplicit(petsc_fisher_multiimplicit):
    """
    Problem class implementing the semi-implicit 2D Gray-Scott reaction-diffusion equation with periodic BC and PETSc
    """

    def __init__(self, problem_params, dtype_u=petsc_vec, dtype_f=petsc_vec_imex):
        """
        Initialization routine

        Args:
            problem_params: custom parameters for the example
            dtype_u: PETSc data type (will be passed to parent class)
            dtype_f: PETSc data type with implicit and explicit parts (will be passed to parent class)
        """

        # invoke super init, passing number of dofs, dtype_u and dtype_f
        super(petsc_fisher_semiimplicit, self).__init__(problem_params=problem_params, dtype_u=dtype_u, dtype_f=dtype_f)

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
        self.A.mult(u, f.impl)
        fa1 = self.init.getVecArray(f.impl)
        fa1[0] = 0
        fa1[-1] = 0

        fa2 = self.init.getVecArray(f.expl)
        xa = self.init.getVecArray(u)
        for i in range(self.xs, self.xe):
            fa2[i] = self.params.lambda0**2 * xa[i] * (1 - xa[i] ** self.params.nu)
        fa2[0] = 0
        fa2[-1] = 0

        return f

    def solve_system(self, rhs, factor, u0, t):
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

        self.ksp.setOperators(self.get_sys_mat(factor))
        self.ksp.solve(rhs, me)

        self.ksp_itercount += self.ksp.getIterationNumber()
        self.ksp_ncalls += 1

        return me
