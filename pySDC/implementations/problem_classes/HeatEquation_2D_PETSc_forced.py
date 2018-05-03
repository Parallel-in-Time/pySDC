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

    class Laplace2D(object):

        def __init__(self, da, nu):
            assert da.getDim() == 2
            self.da = da
            self.localX = da.createLocalVec()
            self.nu = nu

        def mult(self, mat, X, Y):
            self.da.globalToLocal(X, self.localX)
            x = self.da.getVecArray(self.localX)
            y = self.da.getVecArray(Y)
            mx, my = self.da.getSizes()
            hx, hy = [1.0 / (m + 1) for m in [mx, my]]
            (xs, xe), (ys, ye) = self.da.getRanges()
            for j in range(ys, ye):
                for i in range(xs, xe):
                    u = x[i, j]  # center
                    u_e = u_w = u_n = u_s = 0
                    if i > 0:    u_w = x[i - 1, j]  # west
                    if i < mx - 1: u_e = x[i + 1, j]  # east
                    if j > 0:    u_s = x[i, j - 1]  # south
                    if j < my - 1: u_n = x[i, j + 1]  # north
                    u_xx = (u_e - 2 * u + u_w) / hx ** 2
                    u_yy = (u_n - 2 * u + u_s) / hy ** 2
                    y[i, j] = self.nu * (u_xx + u_yy)

    class Heat2D(object):

        def __init__(self, da, factor):
            assert da.getDim() == 2
            self.da = da
            self.localX = da.createLocalVec()
            self.factor = factor

        def mult(self, mat, X, Y):
            self.da.globalToLocal(X, self.localX)
            x = self.da.getVecArray(self.localX)
            y = self.da.getVecArray(Y)
            mx, my = self.da.getSizes()
            hx, hy = [1.0 / (m + 1) for m in [mx, my]]
            (xs, xe), (ys, ye) = self.da.getRanges()
            for j in range(ys, ye):
                for i in range(xs, xe):
                    u = x[i, j]  # center
                    u_e = u_w = u_n = u_s = 0
                    if i > 0:    u_w = x[i - 1, j]  # west
                    if i < mx - 1: u_e = x[i + 1, j]  # east
                    if j > 0:    u_s = x[i, j - 1]  # south
                    if j < my - 1: u_n = x[i, j + 1]  # north
                    u_xx = (u_e - 2 * u + u_w) / hx ** 2
                    u_yy = (u_n - 2 * u + u_s) / hy ** 2
                    y[i, j] = u - self.factor * (u_xx + u_yy)

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
        if len(problem_params['nvars']) != 2:
            raise ProblemError('this is a 2d example, got %s' % problem_params['nvars'])

        da = PETSc.DMDA().create([problem_params['nvars'][0], problem_params['nvars'][1]], stencil_width=1)

        # invoke super init, passing number of dofs, dtype_u and dtype_f
        super(heat2d_petsc_forced, self).__init__(init=da, dtype_u=dtype_u, dtype_f=dtype_f, params=problem_params)

        # compute dx, dy and get local ranges
        self.dx = 1.0 / (self.params.nvars[0] + 1)
        self.dy = 1.0 / (self.params.nvars[1] + 1)
        (self.xs, self.xe), (self.ys, self.ye) = self.init.getRanges()

        # compute discretization matrix A and identity
        self.A = self.__get_A()
        self.H = self.__get_Id()
        # self.Id = self.__get_Id(self.params.nvars, self.params.nu, self.dx, self.dy, self.params.comm)

        # setup solver
        self.ksp = PETSc.KSP()
        self.ksp.create(comm=self.params.comm)
        self.ksp.setType('cg')
        pc = self.ksp.getPC()
        # pc.setType('none')
        self.ksp.setInitialGuessNonzero(True)
        self.ksp.setFromOptions()
        # TODO: fill with data
        self.ksp.setTolerances(rtol=1E-12, atol=1E-12, divtol=None, max_it=None)
        # TODO get rid of communicator for nonMPI controllers (purge, then restore)

    def __get_A(self):
        """
        Helper function to assemble PETSc matrix A

        Returns:
            PETSc matrix object
        """
        N = self.params.nvars[0] * self.params.nvars[1]
        A = PETSc.Mat().createPython([N, N], comm=self.init.comm)
        A.setPythonContext(self.Laplace2D(da=self.init, nu=self.params.nu))
        A.setUp()

        return A

    def __get_Id(self):
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

        N = self.params.nvars[0] * self.params.nvars[1]
        A = PETSc.Mat().createPython([N, N], comm=self.init.comm)

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
                fa[i, j] = -np.sin(np.pi * self.params.freq * (i+1) * self.dx) * \
                    np.sin(np.pi * self.params.freq * (j+1) * self.dy) * \
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

        self.H.setPythonContext(self.Heat2D(da=self.init, factor=factor))
        self.H.setUp()

        me = self.dtype_u(u0)
        self.ksp.setOperators(self.H)
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
                xa[i, j] = np.sin(np.pi * self.params.freq * (i+1) * self.dx) * \
                    np.sin(np.pi * self.params.freq * (j+1) * self.dy) * np.cos(t)

        return me
