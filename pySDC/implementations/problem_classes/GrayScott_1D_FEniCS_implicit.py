from __future__ import division
import dolfin as df
import numpy as np
import random
import logging

from pySDC.core.Problem import ptype


class fenics_grayscott(ptype):
    """
    Example implementing the forced 1D heat equation with Dirichlet-0 BC in [0,1]

    Attributes:
        V: function space
        M: mass matrix for FEM
        K: stiffness matrix incl. diffusion coefficient (and correct sign)
        g: forcing term
        bc: boundary conditions
    """

    def __init__(self, cparams, dtype_u, dtype_f):
        """
        Initialization routine

        Args:
            cparams: custom parameters for the example
            dtype_u: particle data type (will be passed parent class)
            dtype_f: acceleration data type (will be passed parent class)
        """

        # define the Dirichlet boundary
        def Boundary(x, on_boundary):
            return on_boundary

        # these parameters will be used later, so assert their existence
        assert 'c_nvars' in cparams
        assert 't0' in cparams
        assert 'family' in cparams
        assert 'order' in cparams
        assert 'refinements' in cparams

        # add parameters as attributes for further reference
        for k, v in cparams.items():
            setattr(self, k, v)

        df.set_log_level(df.WARNING)

        logging.getLogger('FFC').setLevel(logging.WARNING)

        df.parameters["form_compiler"]["optimize"] = True
        df.parameters["form_compiler"]["cpp_optimize"] = True

        # set mesh and refinement (for multilevel)
        mesh = df.IntervalMesh(self.c_nvars, 0, 100)
        for i in range(self.refinements):
            mesh = df.refine(mesh)

        # define function space for future reference
        V = df.FunctionSpace(mesh, self.family, self.order)
        self.V = V * V

        # invoke super init, passing number of dofs, dtype_u and dtype_f
        super(fenics_grayscott, self).__init__(self.V, dtype_u, dtype_f, cparams)

        # rhs in weak form
        self.w = df.Function(self.V)
        q1, q2 = df.TestFunctions(self.V)

        self.w1, self.w2 = df.split(self.w)

        self.F1 = (-self.Du * df.inner(df.nabla_grad(self.w1), df.nabla_grad(q1)) - self.w1 * (
        self.w2 ** 2) * q1 + self.A * (1 - self.w1) * q1) * df.dx
        self.F2 = (-self.Dv * df.inner(df.nabla_grad(self.w2), df.nabla_grad(q2)) + self.w1 * (
        self.w2 ** 2) * q2 - self.B * self.w2 * q2) * df.dx
        self.F = self.F1 + self.F2

        # mass matrix
        u1, u2 = df.TrialFunctions(self.V)
        a_M = u1 * q1 * df.dx
        M1 = df.assemble(a_M)
        a_M = u2 * q2 * df.dx
        M2 = df.assemble(a_M)
        self.M = M1 + M2

        # self.bc = df.DirichletBC(self.V, df.Constant(0.0), Boundary)

    def __invert_mass_matrix(self, u):
        """
        Helper routine to invert mass matrix

        Args:
            u: current values

        Returns:
            inv(M)*u
        """

        me = self.dtype_u(self.V)

        A = 1.0 * self.M
        b = self.dtype_u(u)

        # self.bc.apply(A,b.values.vector())

        df.solve(A, me.values.vector(), b.values.vector())

        return me

    def solve_system(self, rhs, factor, u0, t):
        """
        Dolfin's linear solver for (M-dtA)u = rhs

        Args:
            rhs: right-hand side for the nonlinear system
            factor: abbrev. for the node-to-node stepsize (or any other factor required)
            u0: initial guess for the iterative solver (not used here so far)

        Returns:
            solution as mesh
        """

        sol = self.dtype_u(self.V)

        # self.g.t = t
        self.w.assign(sol.values)

        q1, q2 = df.TestFunctions(self.V)
        w1, w2 = df.split(self.w)
        r1, r2 = df.split(rhs.values)
        F1 = w1 * q1 * df.dx - factor * self.F1 - r1 * q1 * df.dx
        F2 = w2 * q2 * df.dx - factor * self.F2 - r2 * q2 * df.dx
        F = F1 + F2
        du = df.TrialFunction(self.V)
        J = df.derivative(F, self.w, du)

        problem = df.NonlinearVariationalProblem(F, self.w, [], J)
        solver = df.NonlinearVariationalSolver(problem)

        prm = solver.parameters
        prm['newton_solver']['absolute_tolerance'] = 1E-09
        prm['newton_solver']['relative_tolerance'] = 1E-08
        prm['newton_solver']['maximum_iterations'] = 100
        prm['newton_solver']['relaxation_parameter'] = 1.0

        # df.set_log_level(df.PROGRESS)

        solver.solve()

        sol.values.assign(self.w)

        return sol

    def eval_f(self, u, t):
        """
        Routine to evaluate both parts of the RHS

        Args:
            u: current values
            t: current time

        Returns:
            the RHS divided into two parts
        """

        f = self.dtype_f(self.V)

        self.w.assign(u.values)
        f.values = df.Function(self.V, df.assemble(self.F))

        f = self.__invert_mass_matrix(f)

        return f

    def u_exact(self, t):
        """
        Routine to compute the exact solution at time t

        Args:
            t: current time

        Returns:
            exact solution
        """

        class InitialConditions(df.Expression):
            def __init__(self):
                random.seed(2)
                pass

            def eval(self, values, x):
                values[0] = 1 - 0.5 * np.power(np.sin(np.pi * x[0] / 100), 100)
                values[1] = 0.25 * np.power(np.sin(np.pi * x[0] / 100), 100)

            def value_shape(self):
                return (2,)

        uinit = InitialConditions()

        me = self.dtype_u(self.V)
        me.values = df.interpolate(uinit, self.V)

        return me
