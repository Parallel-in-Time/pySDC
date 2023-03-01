import numpy as np
from petsc4py import PETSc

from pySDC.core.Errors import ParameterError, ProblemError
from pySDC.core.Problem import ptype
from pySDC.implementations.datatype_classes.petsc_vec import petsc_vec, petsc_vec_imex


# noinspection PyUnusedLocal
class heat2d_petsc_forced(ptype):
    """
    Example implementing the forced 2D heat equation with Dirichlet BCs in [0,1]^2,
    discretized using central finite differences and realized with PETSc

    Attributes:
        A: second-order FD discretization of the 2D laplace operator
        Id: identity matrix
        dx: distance between two spatial nodes in x direction
        dy: distance between two spatial nodes in y direction
        ksp: PETSc linear solver object
    """
    dtype_u = petsc_vec
    dtype_f = petsc_vec_imex

    def __init__(self,
                 cnvars, nu, freq, refine, 
                 comm=PETSc.COMM_WORLD, sol_tol=1e-10, sol_maxiter=None):
        """
        Initialization routine

        Args:
            problem_params (dict): custom parameters for the example
            dtype_u: PETSc data type (will be passed parent class)
            dtype_f: PETSc data type with implicit and explicit parts (will be passed parent class)
        """
        # make sure parameters have the correct form
        if len(cnvars) != 2:
            raise ProblemError('this is a 2d example, got %s' % cnvars)

        # create DMDA object which will be used for all grid operations
        da = PETSc.DMDA().create(
            [cnvars[0], cnvars[1]], stencil_width=1, comm=comm
        )
        for _ in range(refine):
            da = da.refine()

        # invoke super init, passing number of dofs, dtype_u and dtype_f
        super().__init__(init=da)
        self._makeAttributeAndRegister(
            'cnvars', 'nu', 'freq', 'comm', 'refine', 'comm', 'sol_tol', 'sol_maxiter',
            localVars=locals(), readOnly=True)

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
        Helper function to assemble PETSc matrix A

        Returns:
            PETSc matrix object
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
        Helper function to assemble PETSc identity matrix

        Returns:
            PETSc matrix object
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
        Routine to evaluate the RHS

        Args:
            u (dtype_u): current values
            t (float): current time

        Returns:
            dtype_f: the RHS
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
        """
        KSP linear solver for (I-factor*A)u = rhs

        Args:
            rhs (dtype_f): right-hand side for the linear system
            factor (float): abbrev. for the local stepsize (or any other factor required)
            u0 (dtype_u): initial guess for the iterative solver
            t (float): current time (e.g. for time-dependent BCs)

        Returns:
            dtype_u: solution
        """

        me = self.dtype_u(u0)
        self.ksp.setOperators(self.Id - factor * self.A)
        self.ksp.solve(rhs, me)
        self.ksp_ncalls += 1
        self.ksp_itercount += int(self.ksp.getIterationNumber())
        return me

    def u_exact(self, t):
        """
        Routine to compute the exact solution at time t

        Args:
            t (float): current time

        Returns:
            dtype_u: exact solution
        """

        me = self.dtype_u(self.init)
        xa = self.init.getVecArray(me)
        for i in range(self.xs, self.xe):
            for j in range(self.ys, self.ye):
                xa[i, j] = (
                    np.sin(np.pi * self.freq * i * self.dx)
                    * np.sin(np.pi * self.freq * j * self.dy)
                    * np.cos(t)
                )

        return me
