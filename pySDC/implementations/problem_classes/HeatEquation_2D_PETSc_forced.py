from __future__ import division
import numpy as np
from petsc4py import PETSc

from pySDC.core.Problem import ptype
from pySDC.core.Errors import ParameterError, ProblemError


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

    def __init__(self, problem_params, dtype_u, dtype_f):
        """
        Initialization routine

        Args:
            problem_params (dict): custom parameters for the example
            dtype_u: mesh data type (will be passed parent class)
            dtype_f: mesh data type (will be passed parent class)
        """

        # these parameters will be used later, so assert their existence

        if 'comm' not in problem_params:
            problem_params['comm'] = PETSc.COMM_WORLD
        if 'sol_tol' not in problem_params:
            problem_params['sol_tol'] = 1E-10
        if 'sol_maxiter' not in problem_params:
            problem_params['sol_maxiter'] = None

        essential_keys = ['nvars', 'nu', 'freq', 'comm']
        for key in essential_keys:
            if key not in problem_params:
                msg = 'need %s to instantiate problem, only got %s' % (key, str(problem_params.keys()))
                raise ParameterError(msg)

        # make sure parameters have the correct form
        if len(problem_params['nvars']) != 2:
            raise ProblemError('this is a 2d example, got %s' % problem_params['nvars'])

        # create DMDA object which will be used for all grid operations
        da = PETSc.DMDA().create([problem_params['nvars'][0], problem_params['nvars'][1]], stencil_width=1,
                                 comm=problem_params['comm'])

        # invoke super init, passing number of dofs, dtype_u and dtype_f
        super(heat2d_petsc_forced, self).__init__(init=da, dtype_u=dtype_u, dtype_f=dtype_f, params=problem_params)

        # compute dx, dy and get local ranges
        self.dx = 1.0 / (self.params.nvars[0] - 1)
        self.dy = 1.0 / (self.params.nvars[1] - 1)
        (self.xs, self.xe), (self.ys, self.ye) = self.init.getRanges()

        # compute discretization matrix A and identity
        self.A = self.__get_A()
        self.Id = self.__get_Id()

        # setup solver
        self.ksp = PETSc.KSP()
        self.ksp.create(comm=self.params.comm)
        self.ksp.setType('cg')
        pc = self.ksp.getPC()
        pc.setType('none')
        self.ksp.setInitialGuessNonzero(True)
        self.ksp.setFromOptions()
        self.ksp.setTolerances(rtol=self.params.sol_tol, atol=self.params.sol_tol, max_it=self.params.sol_maxiter)

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
                    diag = self.params.nu * (-2.0 / self.dx ** 2 - 2.0 / self.dy ** 2)
                    for index, value in [
                        ((i, j - 1), self.params.nu / self.dy ** 2),
                        ((i - 1, j), self.params.nu / self.dx ** 2),
                        ((i, j), diag),
                        ((i + 1, j), self.params.nu / self.dx ** 2),
                        ((i, j + 1), self.params.nu / self.dy ** 2),
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
        self.A.mult(u.values, f.impl.values)

        # evaluate forcing term for explicit part
        fa = self.init.getVecArray(f.expl.values)
        for i in range(self.xs, self.xe):
            for j in range(self.ys, self.ye):
                fa[i, j] = -np.sin(np.pi * self.params.freq * i * self.dx) * \
                    np.sin(np.pi * self.params.freq * j * self.dy) * \
                    (np.sin(t) - self.params.nu * 2.0 * (np.pi * self.params.freq) ** 2 * np.cos(t))

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
        self.ksp.solve(rhs.values, me.values)

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
        xa = self.init.getVecArray(me.values)
        for i in range(self.xs, self.xe):
            for j in range(self.ys, self.ye):
                xa[i, j] = np.sin(np.pi * self.params.freq * i * self.dx) * \
                    np.sin(np.pi * self.params.freq * j * self.dy) * np.cos(t)

        return me
