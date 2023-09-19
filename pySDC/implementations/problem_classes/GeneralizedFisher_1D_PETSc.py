import numpy as np
from petsc4py import PETSc

from pySDC.core.Problem import ptype
from pySDC.implementations.datatype_classes.petsc_vec import petsc_vec, petsc_vec_imex, petsc_vec_comp2


class Fisher_full(object):
    """
    Helper class to generate residual and Jacobian matrix for PETSc's nonlinear solver SNES.

    Parameters
    ----------
    da : DMDA object
        Object of PETSc.
    prob : problem instance
        Contains problem information for PETSc.
    factor : float
        Temporal factor (dt*Qd).
    dx : float
        Grid spacing in x direction.

    Attributes
    ----------
    localX : PETSc vector object
        Local vector for PETSc.
    xs, xe : int
        Defines the ranges for spatial domain.
    mx : tuple
        Get sizes for the vector containing the spatial points.
    row : PETSc matrix stencil object
        Row for a matrix.
    col : PETSc matrix stencil object
        Column for a matrix.
    """

    def __init__(self, da, prob, factor, dx):
        """Initialization routine"""
        assert da.getDim() == 1
        self.da = da
        self.factor = factor
        self.dx = dx
        self.prob = prob
        self.localX = da.createLocalVec()
        self.xs, self.xe = self.da.getRanges()[0]
        self.mx = self.da.getSizes()[0]
        self.row = PETSc.Mat.Stencil()
        self.col = PETSc.Mat.Stencil()

    def formFunction(self, snes, X, F):
        """
        Function to evaluate the residual for the Newton solver. This function should be equal to the RHS
        in the solution.

        Parameters
        ----------
        snes : PETSc solver object
            Nonlinear solver object.
        X : PETSc vector object
            Input vector.
        F : PETSc vector object
            Output vector F(X).

        Returns
        -------
        None
            Overwrites F.
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
                f[i] = x[i] - self.factor * (u_xx + self.prob.lambda0**2 * x[i] * (1 - x[i] ** self.prob.nu))

    def formJacobian(self, snes, X, J, P):
        """
        Function to return the Jacobian matrix.

        Parameters
        ----------
        snes : PETSc solver object
            Nonlinear solver object.
        X : PETSc vector object
            Input vector.
        J : PETSc matrix object
            Current Jacobian matrix.
        P : PETSc matrix object
            New Jacobian matrix.

        Returns
        -------
        PETSc.Mat.Structure.SAME_NONZERO_PATTERN
            Matrix status.
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
                    -2.0 / self.dx**2 + self.prob.lambda0**2 * (1.0 - (self.prob.nu + 1) * x[i] ** self.prob.nu)
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
    Helper class to generate residual and Jacobian matrix for PETSc's nonlinear solver SNES.

    Parameters
    ----------
    da : DMDA object
        Object of PETSc.
    prob : problem instance
        Contains problem information for PETSc.
    factor : float
        Temporal factor (dt*Qd).
    dx : float
        Grid spacing in x direction.

    Attributes
    ----------
    localX : PETSc vector object
        Local vector for PETSc.
    """

    def __init__(self, da, prob, factor):
        """Initialization routine"""
        assert da.getDim() == 1
        self.da = da
        self.prob = prob
        self.factor = factor
        self.localX = da.createLocalVec()

    def formFunction(self, snes, X, F):
        """
        Function to evaluate the residual for the Newton solver. This function should be equal to the RHS
        in the solution.

        Parameters
        ----------
        snes : PETSc solver object
            Nonlinear solver object.
        X : PETSc vector object
            Input vector.
        F : PETSc vector object
            Output vector F(X).

        Returns
        -------
        None
            Overwrites F.
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
                f[i] = x[i] - self.factor * self.prob.lambda0**2 * x[i] * (1 - x[i] ** self.prob.nu)

    def formJacobian(self, snes, X, J, P):
        """
        Function to return the Jacobian matrix.

        Parameters
        ----------
        snes : PETSc solver object
            Nonlinear solver object.
        X : PETSc vector object
            Input vector.
        J : PETSc matrix object
            Current Jacobian matrix.
        P : PETSc matrix object
            New Jacobian matrix.

        Returns
        -------
        PETSc.Mat.Structure.SAME_NONZERO_PATTERN
            Matrix status.
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
                diag = 1.0 - self.factor * self.prob.lambda0**2 * (1.0 - (self.prob.nu + 1) * x[i] ** self.prob.nu)
                P.setValueStencil(row, row, diag)
        P.assemble()
        if J != P:
            J.assemble()  # matrix-free operator
        return PETSc.Mat.Structure.SAME_NONZERO_PATTERN


class petsc_fisher_multiimplicit(ptype):
    r"""
    The following one-dimensional problem is an example of a reaction-diffusion equation with traveling waves, and can
    be seen as a generalized Fisher equation. This class implements a special case of the Kolmogorov-Petrovskii-Piskunov
    problem [1]_ using periodic boundary conditions

    .. math::
        \frac{\partial u}{\partial t} = \Delta u + \lambda_0^2 u (1 - u^\nu)

    with exact solution

    u(x, 0) = \left[
            1 + \left(2^{\nu / 2} - 1\right) \exp\left(-(\nu / 2)\sigma_1 x + 2 \lambda_1 t\right)
        \right]^{-2 / \nu}

    for :math:`x \in \mathbb{R}`, and

    .. math::
        \sigma_1 = \lambda_1 - \sqrt{\lambda_1^2 - \lambda_0^2},

    .. math::
        \lambda_1 = \frac{\lambda_0}{2} \left[
            \left(1 + \frac{\nu}{2}\right)^{1/2} + \left(1 + \frac{\nu}{2}\right)^{-1/2}
        \right].

    This class is implemented to be solved in spatial using PETSc. For time-stepping, the problem will be solved
    *multi-implicitly*.

    Parameters
    ----------
    nvars : int
        Spatial resolution.
    lambda0 : float
        Problem parameter : math:`\lambda_0`.
    nu : float
        Problem parameter :math:`\nu`.
    interval : tuple
        Defines the spatial domain.
    comm : PETSc.COMM_WORLD, optional
        Communicator for PETSc.
    lsol_tol : float, optional
        Tolerance for linear solver to terminate.
    nlsol_tol : float, optional
        Tolerance for nonlinear solver to terminate.
    lsol_maxiter : int, optional
        Maximum number of iterations for linear solver.
    nlsol_maxiter : int, optional
        Maximum number of iterations for nonlinear solver.

    Attributes
    ----------
    dx : float
        Mesh grid width.
    xs, xe : int
        Define the ranges.
    A : PETSc matrix object
        Discretization matrix.
    localX : PETSc vector object
        Local vector for solution.
    ksp : PETSc solver object
        Linear solver object.
    snes : PETSc solver object
        Nonlinear solver object.
    F : PETSc vector object
        Global vector.
    J : PETSc matrix object
        Jacobi matrix.

    References
    ----------
    .. [1] Z. Feng. Traveling wave behavior for a generalized fisher equation. Chaos Solitons Fractals 38(2),
        481–488 (2008).
    """

    dtype_u = petsc_vec
    dtype_f = petsc_vec_comp2

    def __init__(
        self,
        nvars=127,
        lambda0=2.0,
        nu=1.0,
        interval=(-5, 5),
        comm=PETSc.COMM_WORLD,
        lsol_tol=1e-10,
        nlsol_tol=1e-10,
        lsol_maxiter=None,
        nlsol_maxiter=None,
    ):
        """Initialization routine"""
        # create DMDA object which will be used for all grid operations
        da = PETSc.DMDA().create([nvars], dof=1, stencil_width=1, comm=comm)

        # invoke super init, passing number of dofs, dtype_u and dtype_f
        super().__init__(init=da)
        self._makeAttributeAndRegister(
            'nvars',
            'lambda0',
            'nu',
            'interval',
            'comm',
            'lsol_tol',
            'nlsol_tol',
            'lsol_maxiter',
            'nlsol_maxiter',
            localVars=locals(),
            readOnly=True,
        )

        # compute dx and get local ranges
        self.dx = (self.interval[1] - self.interval[0]) / (self.nvars - 1)
        (self.xs, self.xe) = self.init.getRanges()[0]

        # compute discretization matrix A and identity
        self.A = self.__get_A()
        self.localX = self.init.createLocalVec()

        # setup linear solver
        self.ksp = PETSc.KSP()
        self.ksp.create(comm=self.comm)
        self.ksp.setType('cg')
        pc = self.ksp.getPC()
        pc.setType('ilu')
        self.ksp.setInitialGuessNonzero(True)
        self.ksp.setFromOptions()
        self.ksp.setTolerances(rtol=self.lsol_tol, atol=self.lsol_tol, max_it=self.lsol_maxiter)
        self.ksp_itercount = 0
        self.ksp_ncalls = 0

        # setup nonlinear solver
        self.snes = PETSc.SNES()
        self.snes.create(comm=self.comm)
        if self.nlsol_maxiter <= 1:
            self.snes.setType('ksponly')
        self.snes.getKSP().setType('cg')
        pc = self.snes.getKSP().getPC()
        pc.setType('ilu')
        # self.snes.setType('ngmres')
        self.snes.setFromOptions()
        self.snes.setTolerances(
            rtol=self.nlsol_tol,
            atol=self.nlsol_tol,
            stol=self.nlsol_tol,
            max_it=self.nlsol_maxiter,
        )
        self.snes_itercount = 0
        self.snes_ncalls = 0
        self.F = self.init.createGlobalVec()
        self.J = self.init.createMatrix()

    def __get_A(self):
        """
        Helper function to assemble PETSc matrix A.

        Returns
        -------
        A : PETSc matrix object
            Discretization matrix.
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
        Helper function to assemble the system matrix of the linear problem.

        Parameters
        ----------
        factor : float
            Factor to define the system matrix.

        Returns
        -------
        A : PETSc matrix object
           Matrix for the system to solve. 
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
        Routine to evaluate the right-hand side of the problem.

        Parameters
        ----------
        u : dtype_u
            Current values of the numerical solution.

        t : float
            Current time the numerical solution is computed.

        Returns
        -------
        f : dtype_f
            Right-hand side of the problem.
        """

        f = self.dtype_f(self.init)
        self.A.mult(u, f.comp1)
        fa1 = self.init.getVecArray(f.comp1)
        fa1[0] = 0
        fa1[-1] = 0

        fa2 = self.init.getVecArray(f.comp2)
        xa = self.init.getVecArray(u)
        for i in range(self.xs, self.xe):
            fa2[i] = self.lambda0**2 * xa[i] * (1 - xa[i] ** self.nu)
        fa2[0] = 0
        fa2[-1] = 0

        return f

    def solve_system_1(self, rhs, factor, u0, t):
        r"""
        Linear solver for (I - factor A)\vec{u} = \vec{rhs}.

        Parameters
        ----------
        rhs : dtype_f
            Right-hand side for the linear system.
        factor : float
            Abbrev. for the local stepsize (or any other factor required).
        u0 : dtype_u
            Initial guess for the iterative solver.
        t : float
            Current time (e.g. for time-dependent BCs).

        Returns
        -------
        me : dtype_u
            Solution as mesh.
        """

        me = self.dtype_u(u0)

        self.ksp.setOperators(self.get_sys_mat(factor))
        self.ksp.solve(rhs, me)

        self.ksp_itercount += self.ksp.getIterationNumber()
        self.ksp_ncalls += 1

        return me

    def solve_system_2(self, rhs, factor, u0, t):
        r"""
        Nonlinear solver for (I - factor F)(\vec{u}) = \{rhs}.

        Parameters
        ----------
        rhs : dtype_f
            Right-hand side for the linear system.
        factor : float
            Abbrev. for the local stepsize (or any other factor required).
        u0 : dtype_u
            Initial guess for the iterative solver.
        t : float
            Current time (e.g. for time-dependent BCs).

        Returns
        -------
        me : dtype_u
            Solution as mesh.
        """

        me = self.dtype_u(u0)
        target = Fisher_reaction(self.init, self, factor)

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
        Routine to compute the exact solution at time t.

        Parameters
        ----------
        t : float
            Time of the exact solution.

        Returns
        -------
        me : dtype_u
            Exact solution.
        """

        lam1 = self.lambda0 / 2.0 * ((self.nu / 2.0 + 1) ** 0.5 + (self.nu / 2.0 + 1) ** (-0.5))
        sig1 = lam1 - np.sqrt(lam1**2 - self.lambda0**2)
        me = self.dtype_u(self.init)
        xa = self.init.getVecArray(me)
        for i in range(self.xs, self.xe):
            xa[i] = (
                1
                + (2 ** (self.nu / 2.0) - 1)
                * np.exp(-self.nu / 2.0 * sig1 * (self.interval[0] + (i + 1) * self.dx + 2 * lam1 * t))
            ) ** (-2.0 / self.nu)

        return me


class petsc_fisher_fullyimplicit(petsc_fisher_multiimplicit):
    r"""
    The following one-dimensional problem is an example of a reaction-diffusion equation with traveling waves, and can
    be seen as a generalized Fisher equation. This class implements a special case of the Kolmogorov-Petrovskii-Piskunov
    problem [1]_ using periodic boundary conditions

    .. math::
        \frac{\partial u}{\partial t} = \Delta u + \lambda_0^2 u (1 - u^\nu)

    with exact solution

    u(x, 0) = \left[
            1 + \left(2^{\nu / 2} - 1\right) \exp\left(-(\nu / 2)\sigma_1 x + 2 \lambda_1 t\right)
        \right]^{-2 / \nu}

    for :math:`x \in \mathbb{R}`, and

    .. math::
        \sigma_1 = \lambda_1 - \sqrt{\lambda_1^2 - \lambda_0^2},

    .. math::
        \lambda_1 = \frac{\lambda_0}{2} \left[
            \left(1 + \frac{\nu}{2}\right)^{1/2} + \left(1 + \frac{\nu}{2}\right)^{-1/2}
        \right].

    This class is implemented to be solved in spatial using PETSc. For time-stepping, the problem is treated *fully-implicitly*.
    """

    dtype_f = petsc_vec

    def eval_f(self, u, t):
        """
        Routine to evaluate the right-hand side of the problem.

        Parameters
        ----------
        u : dtype_u
            Current values of the numerical solution.
        t : float
            Current time the numerical solution is computed.

        Returns
        -------
        f : dtype_f
            Right-hand side of the problem.
        """

        f = self.dtype_f(self.init)
        self.A.mult(u, f)

        fa2 = self.init.getVecArray(f)
        xa = self.init.getVecArray(u)
        for i in range(self.xs, self.xe):
            fa2[i] += self.lambda0**2 * xa[i] * (1 - xa[i] ** self.nu)
        fa2[0] = 0
        fa2[-1] = 0

        return f

    def solve_system(self, rhs, factor, u0, t):
        r"""
        Nonlinear solver for (I - factor F)(\vec{u}) = \{rhs}.

        Parameters
        ----------
        rhs : dtype_f
            Right-hand side for the linear system.
        factor : float
            Abbrev. for the local stepsize (or any other factor required).
        u0 : dtype_u
            Initial guess for the iterative solver.
        t : float
            Current time (e.g. for time-dependent BCs).

        Returns
        -------
        me : dtype_u
            Solution as mesh.
        """

        me = self.dtype_u(u0)
        target = Fisher_full(self.init, self, factor, self.dx)

        # assign residual function and Jacobian

        self.snes.setFunction(target.formFunction, self.F)
        self.snes.setJacobian(target.formJacobian, self.J)

        self.snes.solve(rhs, me)

        self.snes_itercount += self.snes.getIterationNumber()
        self.snes_ncalls += 1

        return me


class petsc_fisher_semiimplicit(petsc_fisher_multiimplicit):
    r"""
    The following one-dimensional problem is an example of a reaction-diffusion equation with traveling waves, and can
    be seen as a generalized Fisher equation. This class implements a special case of the Kolmogorov-Petrovskii-Piskunov
    problem [1]_ using periodic boundary conditions

    .. math::
        \frac{\partial u}{\partial t} = \Delta u + \lambda_0^2 u (1 - u^\nu)

    with exact solution

    u(x, 0) = \left[
            1 + \left(2^{\nu / 2} - 1\right) \exp\left(-(\nu / 2)\sigma_1 x + 2 \lambda_1 t\right)
        \right]^{-2 / \nu}

    for :math:`x \in \mathbb{R}`, and

    .. math::
        \sigma_1 = \lambda_1 - \sqrt{\lambda_1^2 - \lambda_0^2},

    .. math::
        \lambda_1 = \frac{\lambda_0}{2} \left[
            \left(1 + \frac{\nu}{2}\right)^{1/2} + \left(1 + \frac{\nu}{2}\right)^{-1/2}
        \right].

    This class is implemented to be solved in spatial using PETSc. For time-stepping, the problem here will be solved in
    a *semi-implicit* way.
    """

    dtype_f = petsc_vec_imex

    def eval_f(self, u, t):
        """
        Routine to evaluate the right-hand side of the problem.

        Parameters
        ----------
        u : dtype_u
            Current values of the numerical solution.
        t : float
            Current time the numerical solution is computed.

        Returns
        -------
        f : dtype_f
            Right-hand side of the problem.
        """

        f = self.dtype_f(self.init)
        self.A.mult(u, f.impl)
        fa1 = self.init.getVecArray(f.impl)
        fa1[0] = 0
        fa1[-1] = 0

        fa2 = self.init.getVecArray(f.expl)
        xa = self.init.getVecArray(u)
        for i in range(self.xs, self.xe):
            fa2[i] = self.lambda0**2 * xa[i] * (1 - xa[i] ** self.nu)
        fa2[0] = 0
        fa2[-1] = 0

        return f

    def solve_system(self, rhs, factor, u0, t):
        r"""
        Simple linear solver for (I-factor A)\vec{u} = \{rhs}.

        Parameters
        ----------
        rhs : dtype_f
            Right-hand side for the linear system.
        factor : float
            Abbrev. for the local stepsize (or any other factor required).
        u0 : dtype_u
            Initial guess for the iterative solver.
        t : float
            Current time (e.g. for time-dependent BCs).

        Returns
        -------
        me : dtype_u
            Solution as mesh.
        """

        me = self.dtype_u(u0)

        self.ksp.setOperators(self.get_sys_mat(factor))
        self.ksp.solve(rhs, me)

        self.ksp_itercount += self.ksp.getIterationNumber()
        self.ksp_ncalls += 1

        return me
