from __future__ import division
import dolfin as df
import numpy as np
import random
from petsc4py import PETSc

from pySDC.core.Problem import ptype
from pySDC.core.Errors import ParameterError


# noinspection PyUnusedLocal
class petsc_grayscott(ptype):
    """

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
        def Boundary(x, on_boundary):
            return on_boundary

        # these parameters will be used later, so assert their existence
        essential_keys = ['nvars', 'Du', 'Dv', 'A', 'B']
        for key in essential_keys:
            if key not in problem_params:
                msg = 'need %s to instantiate problem, only got %s' % (key, str(problem_params.keys()))
                raise ParameterError(msg)

        da = PETSc.DMDA().create([problem_params['nvars'][0], problem_params['nvars'][1]], stencil_width=1,
                                 comm=problem_params['comm'])

        # invoke super init, passing number of dofs, dtype_u and dtype_f
        super(petsc_grayscott, self).__init__(init=da, dtype_u=dtype_u, dtype_f=dtype_f, params=problem_params)


       # compute dx, dy and get local ranges
        self.dx = 1.0 / (self.params.nvars[0] - 1)
        self.dy = 1.0 / (self.params.nvars[1] - 1)
        (self.xs, self.xe), (self.ys, self.ye) = self.init.getRanges()

        # compute discretization matrix A and identity
        self.A = self.__get_A()
        self.Id = self.__get_Id()
        self.localX = self.init.createLocalVec()

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
                if i == 0 or j == 0 or i == mx - 1 or j == my - 1:
                    A.setValueStencil(row, row, 1.0)
                else:
                    diag = self.params.Du * (-2.0 / self.dx ** 2 - 2.0 / self.dy ** 2)
                    for index, value in [
                        ((i, j - 1), self.params.Du / self.dy ** 2),
                        ((i - 1, j), self.params.Du / self.dx ** 2),
                        ((i, j), diag),
                        ((i + 1, j), self.params.Du / self.dx ** 2),
                        ((i, j + 1), self.params.Du / self.dy ** 2),
                    ]:
                        col.index = index
                        col.field = 0
                        A.setValueStencil(row, col, value)
                row.index = (i, j)
                row.field = 1
                if i == 0 or j == 0 or i == mx - 1 or j == my - 1:
                    A.setValueStencil(row, row, 1.0)
                else:
                    diag = self.params.Dv * (-2.0 / self.dx ** 2 - 2.0 / self.dy ** 2)
                    for index, value in [
                        ((i, j - 1), self.params.Dv / self.dy ** 2),
                        ((i - 1, j), self.params.Dv / self.dx ** 2),
                        ((i, j), diag),
                        ((i + 1, j), self.params.Dv / self.dx ** 2),
                        ((i, j + 1), self.params.Dv / self.dy ** 2),
                    ]:
                        col.index = index
                        col.field = 1
                        A.setValueStencil(row, col, value)
        A.assemble()

        return A

    def __get_Id(self):
        """
        Helper function to assemble PETSc identity matrix

        Returns:
            PETSc matrix object
        """

        Id = self.init.createMatrix()
        Id.setType('aij')  # sparse
        Id.setFromOptions()
        Id.setPreallocationNNZ((5, 5))
        Id.setUp()

        Id.zeroEntries()
        row = PETSc.Mat.Stencil()
        col = PETSc.Mat.Stencil()
        mx, my = self.init.getSizes()
        (xs, xe), (ys, ye) = self.init.getRanges()
        for j in range(ys, ye):
            for i in range(xs, xe):
                for indx in [0, 1]:
                    row.index = (i, j)
                    row.field = indx
                    Id.setValueStencil(row, col, 1.0)

        Id.assemble()

        return Id

    def __form_Jacobian(self, sens, X, J, P):
        self.init.globalToLocal(X, self.localX)
        x = self.init.getVecArray(self.localX)
        mx, my = self.init.getSizes()
        P.zeroEntries()
        row = PETSc.Mat.Stencil()
        col = PETSc.Mat.Stencil()
        (xs, xe), (ys, ye) = self.init.getRanges()
        for j in range(ys, ye):
            for i in range(xs, xe):
                row.index = (i,j)
                row.field = 0
                if (i==0    or j==0    or k==0 or
                    i==mx-1 or j==my-1 or k==mz-1):
                    P.setValueStencil(row, row, 1.0)
                else:
                    u = x[i,j,k]
                    diag = (2*(hyhzdhx+hxhzdhy+hxhydhz)
                            - lambda_*exp(u)*hxhyhz)
                    for index, value in [
                        ((i,j,k-1), -hxhydhz),
                        ((i,j-1,k), -hxhzdhy),
                        ((i-1,j,k), -hyhzdhx),
                        ((i, j, k), diag),
                        ((i+1,j,k), -hyhzdhx),
                        ((i,j+1,k), -hxhzdhy),
                        ((i,j,k+1), -hxhydhz),
                        ]:
                        col.index = index
                        col.field = 0
                        P.setValueStencil(row, col, value)
        P.assemble()
        if J != P: J.assemble() # matrix-free operator
        return PETSc.Mat.Structure.SAME_NONZERO_PATTERN

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
                fa[i, j] = -np.sin(np.pi * self.params.freq * i * self.dx) * \
                    np.sin(np.pi * self.params.freq * j * self.dy) * \
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
                xa[i, j, 0] = 1.0 - 0.5 * np.power(np.sin(np.pi * i * self.dx / 100) *
                                                   np.sin(np.pi * j * self.dy / 100), 100)
                xa[i, j, 1] = 0.25 * np.power(np.sin(np.pi * i * self.dx / 100) *
                                              np.sin(np.pi * j * self.dy / 100), 100)

        return me
