import numpy as np
from petsc4py import PETSc

from pySDC.core.Errors import ParameterError, ProblemError
from pySDC.core.Problem import ptype
from pySDC.implementations.datatype_classes.petsc_vec import petsc_vec, petsc_vec_imex


# noinspection PyUnusedLocal
class heat2d_petsc_forced(ptype):
    r"""
    Example implementing the forced two-dimensional heat equation with Dirichlet boundary conditions
    :math:`(x, y) \in [0,1]^2`

    .. math::
        \frac{\partial u}{\partial t} = \nu
        \left(
            \frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2}
        \right) + f(x, y, t)

    and forcing term :math:`f(x, y, t)` such that the exact solution is

    .. math::
        u(x, y, t) = \sin(2 \pi x) \sin(2 \pi y) \cos(t).

    The spatial discretization uses central finite differences and is realized with PETSc [1]_, [2]_.

    Parameters
    ----------
    cnvars : tuple, optional
        Spatial resolution for the 2D problem, e.g. (16, 16).
    nu : float, optional
        Diffusion coefficient :math:`\nu`.
    freq : int, optional
        Spatial frequency of the initial conditions (equal for both dimensions).
    refine : int, optional
        Defines the refinement of the mesh, e.g. refine=2 means the mesh is refined with factor 2.
    comm : COMM_WORLD
        Communicator for PETSc.
    sol_tol : float, optional
        Tolerance that the solver needs to satisfy for termination.
    sol_maxiter : int, optional
        Maximum number of iterations for the solver to terminate.

    Attributes
    ----------
    A : PETSc matrix object
        Second-order FD discretization of the 2D Laplace operator.
    Id : PETSc matrix object
        Identity matrix.
    dx : float
        Distance between two spatial nodes in x direction.
    dy : float
        Distance between two spatial nodes in y direction.
    ksp : object
        PETSc linear solver object.
    ksp_ncalls : int
        Calls of PETSc's linear solver object.
    ksp_itercount : int
        Iterations done by PETSc's linear solver object.

    References
    ----------
    .. [1] PETSc Web page. Satish Balay et al. https://petsc.org/ (2023).
    .. [2] Parallel distributed computing using Python. Lisandro D. Dalcin, Rodrigo R. Paz, Pablo A. Kler,
        Alejandro Cosimo. Advances in Water Resources (2011).
    """

    dtype_u = petsc_vec
    dtype_f = petsc_vec_imex

    def __init__(self, cnvars, nu, freq, refine, comm=PETSc.COMM_WORLD, sol_tol=1e-10, sol_maxiter=None):
        """Initialization routine"""
        # make sure parameters have the correct form
        if len(cnvars) != 2:
            raise ProblemError('this is a 2d example, got %s' % cnvars)

        # create DMDA object which will be used for all grid operations
        da = PETSc.DMDA().create([cnvars[0], cnvars[1]], stencil_width=1, comm=comm)
        for _ in range(refine):
            da = da.refine()

        # invoke super init, passing number of dofs, dtype_u and dtype_f
        super().__init__(init=da)
        self._makeAttributeAndRegister(
            'cnvars',
            'nu',
            'freq',
            'comm',
            'refine',
            'comm',
            'sol_tol',
            'sol_maxiter',
            localVars=locals(),
            readOnly=True,
        )

        # compute dx, dy and get local ranges
        self.dx = 1.0 / (self.init.getSizes()[0] - 1)
        self.dy = 1.0 / (self.init.getSizes()[1] - 1)
        (self.xs, self.xe), (self.ys, self.ye) = self.init.getRanges()

        # compute discretization matrix A and identity
        self.A = self.__get_A()
        self.Id = self.__get_Id()

        # setup solver
        self.ksp = PETSc.KSP()
        self.ksp.create(comm=self.comm)
        self.ksp.setType('gmres')
        pc = self.ksp.getPC()
        pc.setType('none')
        # pc.setType('hypre')
        # self.ksp.setInitialGuessNonzero(True)
        self.ksp.setFromOptions()
        self.ksp.setTolerances(rtol=self.sol_tol, atol=self.sol_tol, max_it=self.sol_maxiter)

        self.ksp_ncalls = 0
        self.ksp_itercount = 0

    def __get_A(self):
        """
        Helper function to assemble PETSc matrix A.

        Returns
        -------
        A : PETSc matrix object
            Matrix A defining the 2D Laplace operator.
        """
        # create matrix and set basic options
        A = self.init.createMatrix()
        A.setType('aij')  # sparse
        A.setFromOptions()
        A.setPreallocationNNZ((5, 5))
        A.setUp()

        # fill matrix
        A.zeroEntries()
        row = PETSc.Mat.Stencil()
        col = PETSc.Mat.Stencil()
        mx, my = self.init.getSizes()
        (xs, xe), (ys, ye) = self.init.getRanges()
        for j in range(ys, ye):
            for i in range(xs, xe):
                row.index = (i, j)
                row.field = 0
                if i == 0 or j == 0 or i == mx - 1 or j == my - 1:
                    A.setValueStencil(row, row, 1.0)
                else:
                    diag = self.nu * (-2.0 / self.dx**2 - 2.0 / self.dy**2)
                    for index, value in [
                        ((i, j - 1), self.nu / self.dy**2),
                        ((i - 1, j), self.nu / self.dx**2),
                        ((i, j), diag),
                        ((i + 1, j), self.nu / self.dx**2),
                        ((i, j + 1), self.nu / self.dy**2),
                    ]:
                        col.index = index
                        col.field = 0
                        A.setValueStencil(row, col, value)
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

        # create matrix and set basic options
        Id = self.init.createMatrix()
        Id.setType('aij')  # sparse
        Id.setFromOptions()
        Id.setPreallocationNNZ((1, 1))
        Id.setUp()

        # fill matrix
        Id.zeroEntries()
        row = PETSc.Mat.Stencil()
        (xs, xe), (ys, ye) = self.init.getRanges()
        for j in range(ys, ye):
            for i in range(xs, xe):
                row.index = (i, j)
                row.field = 0
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
            Current time at which the numerical solution is computed at.

        Returns
        -------
        f : dtype_f
            Right-hand side of the problem.
        """

        f = self.dtype_f(self.init)
        # evaluate Au for implicit part
        self.A.mult(u, f.impl)

        # evaluate forcing term for explicit part
        fa = self.init.getVecArray(f.expl)
        xv, yv = np.meshgrid(range(self.xs, self.xe), range(self.ys, self.ye), indexing='ij')
        fa[self.xs : self.xe, self.ys : self.ye] = (
            -np.sin(np.pi * self.freq * xv * self.dx)
            * np.sin(np.pi * self.freq * yv * self.dy)
            * (np.sin(t) - self.nu * 2.0 * (np.pi * self.freq) ** 2 * np.cos(t))
        )

        return f

    def solve_system(self, rhs, factor, u0, t):
        r"""
        KSP linear solver for :math:`(I - factor \cdot A) \vec{u} = \vec{rhs}`.

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
            Solution.
        """

        me = self.dtype_u(u0)
        self.ksp.setOperators(self.Id - factor * self.A)
        self.ksp.solve(rhs, me)
        self.ksp_ncalls += 1
        self.ksp_itercount += int(self.ksp.getIterationNumber())
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

        me = self.dtype_u(self.init)
        xa = self.init.getVecArray(me)
        for i in range(self.xs, self.xe):
            for j in range(self.ys, self.ye):
                xa[i, j] = np.sin(np.pi * self.freq * i * self.dx) * np.sin(np.pi * self.freq * j * self.dy) * np.cos(t)

        return me
