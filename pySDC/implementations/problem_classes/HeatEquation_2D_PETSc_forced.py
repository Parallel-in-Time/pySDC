from __future__ import division

import numpy as np
import scipy.sparse as sp

from petsc4py import PETSc

from pySDC.core.Problem import ptype
from pySDC.core.Errors import ParameterError, ProblemError


# noinspection PyUnusedLocal
class heat2d_petsc_forced(ptype):
    """
    Example implementing the unforced 2D heat equation with periodic BCs in [0,1]^2,
    discretized using central finite differences

    Attributes:
        A: second-order FD discretization of the 2D laplace operator
        dx: distance between two spatial nodes (here: being the same in both dimensions)
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

        essential_keys = ['nvars', 'nu', 'freq', 'comm']
        for key in essential_keys:
            if key not in problem_params:
                msg = 'need %s to instantiate problem, only got %s' % (key, str(problem_params.keys()))
                raise ParameterError(msg)

        # make sure parameters have the correct form
        if problem_params['freq'] % 2 != 0:
            raise ProblemError('need even number of frequencies due to periodic BCs')
        if len(problem_params['nvars']) != 2:
            raise ProblemError('this is a 2d example, got %s' % problem_params['nvars'])

        da = PETSc.DMDA().create([problem_params['nvars'][0], problem_params['nvars'][1]], stencil_width=1)

        # invoke super init, passing number of dofs, dtype_u and dtype_f
        super(heat2d_petsc_forced, self).__init__(init=da, dtype_u=dtype_u, dtype_f=dtype_f, params=problem_params)

        # compute dx (equal in both dimensions) and get discretization matrix A
        # TODO: dx correct?
        self.dx = 1.0 / (self.params.nvars[0] - 1)
        self.dy = 1.0 / (self.params.nvars[1] - 1)
        self.xvalues = np.array([i * self.dx for i in range(self.params.nvars[0])])
        self.yvalues = np.array([i * self.dy for i in range(self.params.nvars[1])])
        (self.xs, self.xe), (self.ys, self.ye) = self.init.getRanges()

        self.A = self.__get_A(self.params.nvars, self.params.nu, self.dx, self.dy, self.params.comm)
        self.Id = self.__get_Id(self.params.nvars, self.params.nu, self.dx, self.dy, self.params.comm)

        self.ksp = PETSc.KSP()
        self.ksp.create(comm=self.params.comm)
        # use conjugate gradients
        self.ksp.setType('cg')
        # and incomplete Cholesky
        self.ksp.getPC().setType('icc')
        self.ksp.setInitialGuessNonzero(True)
        self.ksp.setFromOptions()
        # TODO: fill with data
        # self.ksp.setTolerances(self, rtol=None, atol=None, divtol=None, max_it=None)


    def __get_A(self, N, nu, dx, dy, comm):
        """
        Helper function to assemble PETSc matrix A

        Args:
            N (list): number of dofs
            nu (float): diffusion coefficient
            dx (float): distance between two spatial nodes in x direction
            dx (float): distance between two spatial nodes in y direction

        Returns:
            scipy.sparse.csc_matrix: matrix A in CSC format
        """

        A = PETSc.Mat()
        A.create(comm=comm)
        A.setSizes([N[0] * N[1], N[0] * N[1]])
        A.setType('aij')  # sparse
        A.setPreallocationNNZ(5)

        diagv = nu * (2.0 / dx ** 2 + 2.0 / dy ** 2)
        offdx = nu * (-1.0 / dx ** 2)
        offdy = nu * (-1.0 / dy ** 2)

        Istart, Iend = A.getOwnershipRange()
        for I in range(Istart, Iend):
            A[I, I] = diagv
            i = I // N[0]  # map row number to
            j = I - i * N[0]  # grid coordinates
            if i > 0:
                J = I - N[0]
                A[I, J] = offdx
            if i < N[1] - 1:
                J = I + N[0]
                A[I, J] = offdx
            if j > 0:
                J = I - 1
                A[I, J] = offdy
            if j < N[0] - 1:
                J = I + 1
                A[I, J] = offdy

        # communicate off-processor values
        # and setup internal data structures
        # for performing parallel operations
        A.assemblyBegin()
        A.assemblyEnd()

        return A

    def __get_Id(self, N, nu, dx, dy, comm):
        """
        Helper function to assemble PETSc matrix A

        Args:
            N (list): number of dofs
            nu (float): diffusion coefficient
            dx (float): distance between two spatial nodes in x direction
            dx (float): distance between two spatial nodes in y direction

        Returns:
            scipy.sparse.csc_matrix: matrix A in CSC format
        """

        A = PETSc.Mat()
        A.create(comm=comm)
        A.setSizes([N[0] * N[1], N[0] * N[1]])
        A.setType('aij')  # sparse
        A.setPreallocationNNZ(5)

        diagv = 1.0

        Istart, Iend = A.getOwnershipRange()
        for I in range(Istart, Iend):
            A[I, I] = diagv

        # communicate off-processor values
        # and setup internal data structures
        # for performing parallel operations
        A.assemblyBegin()
        A.assemblyEnd()

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
        self.A.mult(u.values, f.impl.values)
        fa = self.init.getVecArray(f.expl.values)

        for i in range(self.xs, self.xe):
            for j in range(self.ys, self.ye):
                fa[i, j] = -np.sin(np.pi * self.params.freq * self.xvalues[i]) * \
                    np.sin(np.pi * self.params.freq * self.yvalues[j]) * \
                    (np.sin(t) - self.params.nu * 2.0 * (np.pi * self.params.freq) ** 2 * np.cos(t))

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
                xa[i, j] = np.sin(np.pi * self.params.freq * self.xvalues[i]) * \
                           np.sin(np.pi * self.params.freq * self.yvalues[j]) * np.cos(t)

        return me
