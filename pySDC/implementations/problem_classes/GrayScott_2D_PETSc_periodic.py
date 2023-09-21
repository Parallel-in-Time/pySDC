import numpy as np
from petsc4py import PETSc

from pySDC.core.Problem import ptype
from pySDC.implementations.datatype_classes.petsc_vec import petsc_vec, petsc_vec_imex, petsc_vec_comp2


class GS_full(object):
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
    dy : float
        Grid spacing in y direction.

    Attributes
    ----------
    localX : PETSc vector object
        Local vector for PETSc.
    """

    def __init__(self, da, prob, factor, dx, dy):
        """Initialization routine"""
        assert da.getDim() == 2
        self.da = da
        self.prob = prob
        self.factor = factor
        self.dx = dx
        self.dy = dy
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
        (xs, xe), (ys, ye) = self.da.getRanges()
        for j in range(ys, ye):
            for i in range(xs, xe):
                u = x[i, j]  # center
                u_e = x[i + 1, j]  # east
                u_w = x[i - 1, j]  # west
                u_s = x[i, j + 1]  # south
                u_n = x[i, j - 1]  # north
                u_xx = u_e - 2 * u + u_w
                u_yy = u_n - 2 * u + u_s
                f[i, j, 0] = x[i, j, 0] - (
                    self.factor
                    * (
                        self.prob.Du * (u_xx[0] / self.dx**2 + u_yy[0] / self.dy**2)
                        - x[i, j, 0] * x[i, j, 1] ** 2
                        + self.prob.A * (1 - x[i, j, 0])
                    )
                )
                f[i, j, 1] = x[i, j, 1] - (
                    self.factor
                    * (
                        self.prob.Dv * (u_xx[1] / self.dx**2 + u_yy[1] / self.dy**2)
                        + x[i, j, 0] * x[i, j, 1] ** 2
                        - self.prob.B * x[i, j, 1]
                    )
                )

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
        col = PETSc.Mat.Stencil()
        (xs, xe), (ys, ye) = self.da.getRanges()

        for j in range(ys, ye):
            for i in range(xs, xe):
                # diagnoal 2-by-2 block (for u and v)
                row.index = (i, j)
                col.index = (i, j)
                row.field = 0
                col.field = 0
                val = 1.0 - self.factor * (
                    self.prob.Du * (-2.0 / self.dx**2 - 2.0 / self.dy**2) - x[i, j, 1] ** 2 - self.prob.A
                )
                P.setValueStencil(row, col, val)
                row.field = 0
                col.field = 1
                val = self.factor * 2.0 * x[i, j, 0] * x[i, j, 1]
                P.setValueStencil(row, col, val)
                row.field = 1
                col.field = 1
                val = 1.0 - self.factor * (
                    self.prob.Dv * (-2.0 / self.dx**2 - 2.0 / self.dy**2)
                    + 2.0 * x[i, j, 0] * x[i, j, 1]
                    - self.prob.B
                )
                P.setValueStencil(row, col, val)
                row.field = 1
                col.field = 0
                val = -self.factor * x[i, j, 1] ** 2
                P.setValueStencil(row, col, val)

                # coupling through finite difference part
                col.index = (i, j - 1)
                col.field = 0
                row.field = 0
                P.setValueStencil(row, col, -self.factor * self.prob.Du / self.dx**2)
                col.field = 1
                row.field = 1
                P.setValueStencil(row, col, -self.factor * self.prob.Dv / self.dy**2)
                col.index = (i, j + 1)
                col.field = 0
                row.field = 0
                P.setValueStencil(row, col, -self.factor * self.prob.Du / self.dx**2)
                col.field = 1
                row.field = 1
                P.setValueStencil(row, col, -self.factor * self.prob.Dv / self.dy**2)
                col.index = (i - 1, j)
                col.field = 0
                row.field = 0
                P.setValueStencil(row, col, -self.factor * self.prob.Du / self.dx**2)
                col.field = 1
                row.field = 1
                P.setValueStencil(row, col, -self.factor * self.prob.Dv / self.dy**2)
                col.index = (i + 1, j)
                col.field = 0
                row.field = 0
                P.setValueStencil(row, col, -self.factor * self.prob.Du / self.dx**2)
                col.field = 1
                row.field = 1
                P.setValueStencil(row, col, -self.factor * self.prob.Dv / self.dy**2)

        P.assemble()
        if J != P:
            J.assemble()  # matrix-free operator
        return PETSc.Mat.Structure.SAME_NONZERO_PATTERN


class GS_reaction(object):
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

    Attributes
    ----------
    localX : PETSc vector object
        Local vector for PETSc.
    """

    def __init__(self, da, prob, factor):
        """Initialization routine"""
        assert da.getDim() == 2
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
        (xs, xe), (ys, ye) = self.da.getRanges()
        for j in range(ys, ye):
            for i in range(xs, xe):
                f[i, j, 0] = x[i, j, 0] - (
                    self.factor * (-x[i, j, 0] * x[i, j, 1] ** 2 + self.prob.A * (1 - x[i, j, 0]))
                )
                f[i, j, 1] = x[i, j, 1] - (self.factor * (x[i, j, 0] * x[i, j, 1] ** 2 - self.prob.B * x[i, j, 1]))

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
        col = PETSc.Mat.Stencil()
        (xs, xe), (ys, ye) = self.da.getRanges()
        for j in range(ys, ye):
            for i in range(xs, xe):
                row.index = (i, j)
                col.index = (i, j)
                row.field = 0
                col.field = 0
                P.setValueStencil(row, col, 1.0 - self.factor * (-x[i, j, 1] ** 2 - self.prob.A))
                row.field = 0
                col.field = 1
                P.setValueStencil(row, col, self.factor * 2.0 * x[i, j, 0] * x[i, j, 1])
                row.field = 1
                col.field = 1
                P.setValueStencil(row, col, 1.0 - self.factor * (2.0 * x[i, j, 0] * x[i, j, 1] - self.prob.B))
                row.field = 1
                col.field = 0
                P.setValueStencil(row, col, -self.factor * x[i, j, 1] ** 2)

        P.assemble()
        if J != P:
            J.assemble()  # matrix-free operator
        return PETSc.Mat.Structure.SAME_NONZERO_PATTERN


class petsc_grayscott_multiimplicit(ptype):
    r"""
    The Gray-Scott system [1]_ describes a reaction-diffusion process of two substances :math:`u` and :math:`v`,
    where they diffuse over time. During the reaction :math:`u` is used up with overall decay rate :math:`B`,
    whereas :math:`v` is produced with feed rate :math:`A`. :math:`D_u,\, D_v` are the diffusion rates for
    :math:`u,\, v`. This process is described by the two-dimensional model using periodic boundary conditions

    .. math::
        \frac{\partial u}{\partial t} = D_u \Delta u - u v^2 + A (1 - u),

    .. math::
        \frac{\partial v}{\partial t} = D_v \Delta v + u v^2 - B u

    for :math:`x \in \Omega:=[0, 100]`. The spatial solve of the problem is realized by PETSc [2]_, [3]_. For time-stepping,
    the diffusion part is solved by one of PETSc's linear solver, whereas the reaction part will be solved by a nonlinear
    solver.

    Parameters
    ----------
    nvars : tuple of int, optional
        Spatial resolution, i.e., number of degrees of freedom in space, e.g. (256, 256).
    Du : float, optional
        Diffusion rate for :math:`u`.
    Dv: float, optional
        Diffusion rate for :math:`v`.
    A : float, optional
        Feed rate for :math:`v`.
    B : float, optional
        Overall decay rate for :math:`u`.
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
        Mesh grid width in x direction.
    dy : float
        Mesh grid width in y direction.
    AMat : PETSc matrix object
        Discretization matrix.
    Id : PETSc matrix object
        Identity matrix.
    localX : PETSc vector object
        Local vector for solution.
    ksp : PETSc solver object
        Linear solver object.
    snes : PETSc solver object
        Nonlinear solver object.
    snes_itercount : int
        Number of iterations done by nonlinear solver.
    snes_ncalls : int
        Number of calls of PETSc's nonlinear solver.

    References
    ----------
    .. [1] Autocatalytic reactions in the isothermal, continuous stirred tank reactor: Isolas and other forms
        of multistability. P. Gray, S. K. Scott. Chem. Eng. Sci. 38, 1 (1983).
    .. [2] PETSc Web page. Satish Balay et al. https://petsc.org/ (2023).
    .. [3] Parallel distributed computing using Python. Lisandro D. Dalcin, Rodrigo R. Paz, Pablo A. Kler,
        Alejandro Cosimo. Advances in Water Resources (2011).
    """

    dtype_u = petsc_vec
    dtype_f = petsc_vec_comp2

    def __init__(
        self,
        nvars,
        Du,
        Dv,
        A,
        B,
        comm=PETSc.COMM_WORLD,
        lsol_tol=1e-10,
        nlsol_tol=1e-10,
        lsol_maxiter=None,
        nlsol_maxiter=None,
    ):
        """Initialization routine"""
        # create DMDA object which will be used for all grid operations (boundary_type=3 -> periodic BC)
        da = PETSc.DMDA().create(
            [nvars[0], nvars[1]],
            dof=2,
            boundary_type=3,
            stencil_width=1,
            comm=comm,
        )

        # invoke super init, passing number of dofs, dtype_u and dtype_f
        super().__init__(init=da)
        self._makeAttributeAndRegister(
            'nvars',
            'Du',
            'Dv',
            'A',
            'B',
            'comm',
            'lsol_tol',
            'lsol_maxiter',
            'nlsol_tol',
            'nlsol_maxiter',
            localVars=locals(),
            readOnly=True,
        )

        # compute dx, dy and get local ranges
        self.dx = 100.0 / (self.nvars[0])
        self.dy = 100.0 / (self.nvars[1])
        (self.xs, self.xe), (self.ys, self.ye) = self.init.getRanges()

        # compute discretization matrix A and identity
        self.AMat = self.__get_A()
        self.Id = self.__get_Id()
        self.localX = self.init.createLocalVec()

        # setup linear solver
        self.ksp = PETSc.KSP()
        self.ksp.create(comm=self.comm)
        self.ksp.setType('cg')
        pc = self.ksp.getPC()
        pc.setType('none')
        self.ksp.setInitialGuessNonzero(True)
        self.ksp.setFromOptions()
        self.ksp.setTolerances(rtol=self.lsol_tol, atol=self.lsol_tol, max_it=self.lsol_maxiter)
        self.ksp_itercount = 0
        self.ksp_ncalls = 0

        # setup nonlinear solver
        self.snes = PETSc.SNES()
        self.snes.create(comm=self.comm)
        # self.snes.getKSP().setType('cg')
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

    def __get_A(self):
        """
        Helper function to assemble PETSc matrix A.

        Returns
        -------
        A : PETSc matrix object
            Discretization matrix.
        """
        A = self.init.createMatrix()
        A.setType('aij')  # sparse
        A.setFromOptions()
        A.setPreallocationNNZ((5, 5))
        A.setUp()

        A.zeroEntries()
        row = PETSc.Mat.Stencil()
        col = PETSc.Mat.Stencil()
        mx, my = self.init.getSizes()
        (xs, xe), (ys, ye) = self.init.getRanges()
        for j in range(ys, ye):
            for i in range(xs, xe):
                row.index = (i, j)
                row.field = 0
                A.setValueStencil(row, row, self.Du * (-2.0 / self.dx**2 - 2.0 / self.dy**2))
                row.field = 1
                A.setValueStencil(row, row, self.Dv * (-2.0 / self.dx**2 - 2.0 / self.dy**2))
                # if j > 0:
                col.index = (i, j - 1)
                col.field = 0
                row.field = 0
                A.setValueStencil(row, col, self.Du / self.dy**2)
                col.field = 1
                row.field = 1
                A.setValueStencil(row, col, self.Dv / self.dy**2)
                # if j < my - 1:
                col.index = (i, j + 1)
                col.field = 0
                row.field = 0
                A.setValueStencil(row, col, self.Du / self.dy**2)
                col.field = 1
                row.field = 1
                A.setValueStencil(row, col, self.Dv / self.dy**2)
                # if i > 0:
                col.index = (i - 1, j)
                col.field = 0
                row.field = 0
                A.setValueStencil(row, col, self.Du / self.dx**2)
                col.field = 1
                row.field = 1
                A.setValueStencil(row, col, self.Dv / self.dx**2)
                # if i < mx - 1:
                col.index = (i + 1, j)
                col.field = 0
                row.field = 0
                A.setValueStencil(row, col, self.Du / self.dx**2)
                col.field = 1
                row.field = 1
                A.setValueStencil(row, col, self.Dv / self.dx**2)
        A.assemble()

        return A

    def __get_Id(self):
        """
        Helper function to assemble PETSc identity matrix.

        Returns
        -------
        Id : PETSc matrix object
            Identity matrix.
        """

        Id = self.init.createMatrix()
        Id.setType('aij')  # sparse
        Id.setFromOptions()
        Id.setPreallocationNNZ((1, 1))
        Id.setUp()

        Id.zeroEntries()
        row = PETSc.Mat.Stencil()
        mx, my = self.init.getSizes()
        (xs, xe), (ys, ye) = self.init.getRanges()
        for j in range(ys, ye):
            for i in range(xs, xe):
                for indx in [0, 1]:
                    row.index = (i, j)
                    row.field = indx
                    Id.setValueStencil(row, row, 1.0)

        Id.assemble()

        return Id

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
        self.AMat.mult(u, f.comp1)

        fa = self.init.getVecArray(f.comp2)
        xa = self.init.getVecArray(u)
        for i in range(self.xs, self.xe):
            for j in range(self.ys, self.ye):
                fa[i, j, 0] = -xa[i, j, 0] * xa[i, j, 1] ** 2 + self.A * (1 - xa[i, j, 0])
                fa[i, j, 1] = xa[i, j, 0] * xa[i, j, 1] ** 2 - self.B * xa[i, j, 1]

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
        self.ksp.setOperators(self.Id - factor * self.AMat)
        self.ksp.solve(rhs, me)

        self.ksp_ncalls += 1
        self.ksp_itercount += self.ksp.getIterationNumber()

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
        target = GS_reaction(self.init, self, factor)

        F = self.init.createGlobalVec()
        self.snes.setFunction(target.formFunction, F)
        J = self.init.createMatrix()
        self.snes.setJacobian(target.formJacobian, J)

        self.snes.solve(rhs, me)

        self.snes_ncalls += 1
        self.snes_itercount += self.snes.getIterationNumber()

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

        assert t == 0, 'ERROR: u_exact is only valid for the initial solution'

        me = self.dtype_u(self.init)
        xa = self.init.getVecArray(me)
        for i in range(self.xs, self.xe):
            for j in range(self.ys, self.ye):
                xa[i, j, 0] = 1.0 - 0.5 * np.power(
                    np.sin(np.pi * i * self.dx / 100) * np.sin(np.pi * j * self.dy / 100), 100
                )
                xa[i, j, 1] = 0.25 * np.power(
                    np.sin(np.pi * i * self.dx / 100) * np.sin(np.pi * j * self.dy / 100), 100
                )

        return me


class petsc_grayscott_fullyimplicit(petsc_grayscott_multiimplicit):
    r"""
    The Gray-Scott system [1]_ describes a reaction-diffusion process of two substances :math:`u` and :math:`v`,
    where they diffuse over time. During the reaction :math:`u` is used up with overall decay rate :math:`B`,
    whereas :math:`v` is produced with feed rate :math:`A`. :math:`D_u,\, D_v` are the diffusion rates for
    :math:`u,\, v`. This process is described by the two-dimensional model using periodic boundary conditions

    .. math::
        \frac{\partial u}{\partial t} = D_u \Delta u - u v^2 + A (1 - u),

    .. math::
        \frac{\partial v}{\partial t} = D_v \Delta v + u v^2 - B u

    for :math:`x \in \Omega:=[0, 100]`. The spatial solve of the problem is realized by PETSc [2]_, [3]_. For time-stepping, the
    problem is handled in a *fully-implicit* way, i.e., the nonlinear system containing the full right-hand side will be
    solved by PETSc's nonlinear solver.
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
        self.AMat.mult(u, f)

        fa = self.init.getVecArray(f)
        xa = self.init.getVecArray(u)
        for i in range(self.xs, self.xe):
            for j in range(self.ys, self.ye):
                fa[i, j, 0] += -xa[i, j, 0] * xa[i, j, 1] ** 2 + self.A * (1 - xa[i, j, 0])
                fa[i, j, 1] += xa[i, j, 0] * xa[i, j, 1] ** 2 - self.B * xa[i, j, 1]

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
        target = GS_full(self.init, self, factor, self.dx, self.dy)

        # assign residual function and Jacobian
        F = self.init.createGlobalVec()
        self.snes.setFunction(target.formFunction, F)
        J = self.init.createMatrix()
        self.snes.setJacobian(target.formJacobian, J)

        self.snes.solve(rhs, me)

        self.snes_ncalls += 1
        self.snes_itercount += self.snes.getIterationNumber()

        return me


class petsc_grayscott_semiimplicit(petsc_grayscott_multiimplicit):
    r"""
    The Gray-Scott system [1]_ describes a reaction-diffusion process of two substances :math:`u` and :math:`v`,
    where they diffuse over time. During the reaction :math:`u` is used up with overall decay rate :math:`B`,
    whereas :math:`v` is produced with feed rate :math:`A`. :math:`D_u,\, D_v` are the diffusion rates for
    :math:`u,\, v`. This process is described by the two-dimensional model using periodic boundary conditions

    .. math::
        \frac{\partial u}{\partial t} = D_u \Delta u - u v^2 + A (1 - u),

    .. math::
        \frac{\partial v}{\partial t} = D_v \Delta v + u v^2 - B u

    for :math:`x \in \Omega:=[0, 100]`. The spatial solve of the problem is realized by PETSc [2]_, [3]_. For time-stepping, the
    problem is treated *semi-implicitly*, i.e., the system with diffusion part is solved by PETSc's linear solver.
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
        self.AMat.mult(u, f.impl)

        fa = self.init.getVecArray(f.expl)
        xa = self.init.getVecArray(u)
        for i in range(self.xs, self.xe):
            for j in range(self.ys, self.ye):
                fa[i, j, 0] = -xa[i, j, 0] * xa[i, j, 1] ** 2 + self.A * (1 - xa[i, j, 0])
                fa[i, j, 1] = xa[i, j, 0] * xa[i, j, 1] ** 2 - self.B * xa[i, j, 1]

        return f

    def solve_system(self, rhs, factor, u0, t):
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
        self.ksp.setOperators(self.Id - factor * self.AMat)
        self.ksp.solve(rhs, me)

        self.ksp_ncalls += 1
        self.ksp_itercount += self.ksp.getIterationNumber()

        return me
