import logging

import dolfin as df
import numpy as np

from pySDC.core.Errors import ParameterError
from pySDC.core.Problem import ptype
from pySDC.implementations.datatype_classes.fenics_mesh import fenics_mesh, rhs_fenics_mesh


# noinspection PyUnusedLocal
class fenics_heat_weak_fullyimplicit(ptype):
    """
    Example implementing the forced 1D heat equation with Dirichlet-0 BC in [0,1], weak formulation

    Attributes:
        V: function space
        w: function for weak form
        a_K: weak form of RHS (incl. BC)
        M: mass matrix for FEM
        g: forcing term
        bc: boundary conditions
    """

    def __init__(self, problem_params, dtype_u=fenics_mesh, dtype_f=fenics_mesh):
        """
        Initialization routine

        Args:
            problem_params (dict): custom parameters for the example
            dtype_u: FEniCS mesh data type (will be passed to parent class)
            dtype_f: FEniCS mesh data type (will be passed to parent class)
        """

        # define the Dirichlet boundary
        def Boundary(x, on_boundary):
            return on_boundary

        # these parameters will be used later, so assert their existence
        essential_keys = ['c_nvars', 't0', 'family', 'order', 'refinements', 'nu']
        for key in essential_keys:
            if key not in problem_params:
                msg = 'need %s to instantiate problem, only got %s' % (key, str(problem_params.keys()))
                raise ParameterError(msg)

        # set logger level for FFC and dolfin
        logging.getLogger('ULF').setLevel(logging.WARNING)
        logging.getLogger('FFC').setLevel(logging.WARNING)
        df.set_log_level(30)

        # set solver and form parameters
        df.parameters["form_compiler"]["optimize"] = True
        df.parameters["form_compiler"]["cpp_optimize"] = True

        # set mesh and refinement (for multilevel)
        mesh = df.UnitIntervalMesh(problem_params['c_nvars'])
        for _ in range(problem_params['refinements']):
            mesh = df.refine(mesh)

        # define function space for future reference
        self.V = df.FunctionSpace(mesh, problem_params['family'], problem_params['order'])
        tmp = df.Function(self.V)
        print('DoFs on this level:', len(tmp.vector()[:]))

        # invoke super init, passing number of dofs, dtype_u and dtype_f
        super(fenics_heat_weak_fullyimplicit, self).__init__(self.V, dtype_u, dtype_f, problem_params)

        self.g = df.Expression(
            '-sin(a*x[0]) * (sin(t) - b*a*a*cos(t))',
            a=np.pi,
            b=self.params.nu,
            t=self.params.t0,
            degree=self.params.order,
        )

        # rhs in weak form
        self.w = df.Function(self.V)
        v = df.TestFunction(self.V)
        self.a_K = -self.params.nu * df.inner(df.nabla_grad(self.w), df.nabla_grad(v)) * df.dx + self.g * v * df.dx

        # mass matrix
        u = df.TrialFunction(self.V)
        a_M = u * v * df.dx
        self.M = df.assemble(a_M)

        self.bc = df.DirichletBC(self.V, df.Constant(0.0), Boundary)

    def __invert_mass_matrix(self, u):
        """
        Helper routine to invert mass matrix

        Args:
            u (dtype_u): current values

        Returns:
            dtype_u: inv(M)*u
        """

        me = self.dtype_u(self.V)

        A = 1.0 * self.M
        b = self.dtype_u(u)

        self.bc.apply(A, b.values.vector())

        df.solve(A, me.values.vector(), b.values.vector())

        return me

    def solve_system(self, rhs, factor, u0, t):
        """
        Dolfin's weak solver for (M-dtA)u = rhs

        Args:
            rhs (dtype_f): right-hand side for the nonlinear system
            factor (float): abbrev. for the node-to-node stepsize (or any other factor required)
            u0 (dtype_u_: initial guess for the iterative solver (not used here so far)
            t (float): current time

        Returns:
            dtype_u: solution as mesh
        """

        sol = self.dtype_u(self.V)

        self.g.t = t
        self.w.assign(sol.values)

        v = df.TestFunction(self.V)
        F = self.w * v * df.dx - factor * self.a_K - rhs.values * v * df.dx

        du = df.TrialFunction(self.V)
        J = df.derivative(F, self.w, du)

        problem = df.NonlinearVariationalProblem(F, self.w, self.bc, J)
        solver = df.NonlinearVariationalSolver(problem)

        prm = solver.parameters
        prm['newton_solver']['absolute_tolerance'] = 1e-12
        prm['newton_solver']['relative_tolerance'] = 1e-12
        prm['newton_solver']['maximum_iterations'] = 25
        prm['newton_solver']['relaxation_parameter'] = 1.0

        # df.set_log_level(df.PROGRESS)

        solver.solve()

        sol.values.assign(self.w)

        return sol

    def eval_f(self, u, t):
        """
        Routine to evaluate both parts of the RHS

        Args:
            u (dtype_u): current values
            t (float): current time

        Returns:
            dtype_f: the RHS divided into two parts
        """

        self.g.t = t

        f = self.dtype_f(self.V)

        self.w.assign(u.values)
        f.values = df.Function(self.V, df.assemble(self.a_K))

        f = self.__invert_mass_matrix(f)

        return f

    def u_exact(self, t):
        """
        Routine to compute the exact solution at time t

        Args:
            t (float): current time

        Returns:
            dtype_u: exact solution
        """

        u0 = df.Expression('sin(a*x[0]) * cos(t)', a=np.pi, t=t, degree=self.params.order)
        me = self.dtype_u(self.V)
        me.values = df.interpolate(u0, self.V)

        return me


class fenics_heat_weak_imex(ptype):
    """
    Example implementing the forced 1D heat equation with Dirichlet-0 BC in [0,1], weak formulation, IMEX

    Attributes:
        V: function space
        w: function for weak form
        a_K: weak form of RHS (incl. BC)
        M: mass matrix for FEM
        g: forcing term
        bc: boundary conditions
    """

    def __init__(self, problem_params, dtype_u=fenics_mesh, dtype_f=rhs_fenics_mesh):
        """
        Initialization routine

        Args:
            problem_params (dict): custom parameters for the example
            dtype_u: particle data type (will be passed parent class)
            dtype_f: acceleration data type (will be passed parent class)
        """

        # define the Dirichlet boundary
        def Boundary(x, on_boundary):
            return on_boundary

        # these parameters will be used later, so assert their existence
        essential_keys = ['c_nvars', 't0', 'family', 'order', 'refinements', 'nu']
        for key in essential_keys:
            if key not in problem_params:
                msg = 'need %s to instantiate problem, only got %s' % (key, str(problem_params.keys()))
                raise ParameterError(msg)

        # set logger level for FFC and dolfin
        logging.getLogger('ULF').setLevel(logging.WARNING)
        logging.getLogger('FFC').setLevel(logging.WARNING)
        df.set_log_level(30)

        # set solver and form parameters
        df.parameters["form_compiler"]["optimize"] = True
        df.parameters["form_compiler"]["cpp_optimize"] = True

        # set mesh and refinement (for multilevel)
        mesh = df.UnitIntervalMesh(problem_params['c_nvars'])
        for _ in range(problem_params['refinements']):
            mesh = df.refine(mesh)

        # define function space for future reference
        self.V = df.FunctionSpace(mesh, problem_params['family'], problem_params['order'])
        tmp = df.Function(self.V)
        print('DoFs on this level:', len(tmp.vector()[:]))

        # invoke super init, passing number of dofs, dtype_u and dtype_f
        super(fenics_heat_weak_imex, self).__init__(self.V, dtype_u, dtype_f, problem_params)

        self.g = df.Expression(
            '-sin(a*x[0]) * (sin(t) - b*a*a*cos(t))',
            a=np.pi,
            b=self.params.nu,
            t=self.params.t0,
            degree=self.params.order,
        )

        # rhs in weak form
        self.u = df.TrialFunction(self.V)
        self.v = df.TestFunction(self.V)
        self.a_K = -self.params.nu * df.inner(df.grad(self.u), df.grad(self.v)) * df.dx

        # mass matrix
        a_M = self.u * self.v * df.dx
        self.M = df.assemble(a_M)

        self.bc = df.DirichletBC(self.V, df.Constant(0.0), Boundary)

    def __invert_mass_matrix(self, u):
        """
        Helper routine to invert mass matrix

        Args:
            u (dtype_u): current values

        Returns:
            dtype_u: inv(M)*u
        """

        me = self.dtype_u(self.V)

        b = self.dtype_u(u)

        self.bc.apply(self.M, b.values.vector())

        df.solve(self.M, me.values.vector(), b.values.vector())

        return me

    def solve_system(self, rhs, factor, u0, t):
        """
        Dolfin's weak solver for (M-dtA)u = rhs

        Args:
            rhs (dtype_f): right-hand side for the nonlinear system
            factor (float): abbrev. for the node-to-node stepsize (or any other factor required)
            u0 (dtype_u_: initial guess for the iterative solver (not used here so far)
            t (float): current time

        Returns:
            dtype_u: solution as mesh
        """

        sol = self.dtype_u(u0)

        df.solve(self.u * self.v * df.dx - factor * self.a_K == rhs.values * self.v * df.dx, sol.values, self.bc)

        return sol

    def __eval_fexpl(self, u, t):
        """
        Helper routine to evaluate the explicit part of the RHS

        Args:
            u (dtype_u): current values (not used here)
            t (fliat): current time

        Returns:
            explicit part of RHS
        """

        self.g.t = t
        fexpl = self.dtype_u(df.interpolate(self.g, self.V))

        return fexpl

    def __eval_fimpl(self, u, t):
        """
        Helper routine to evaluate the implicit part of the RHS

        Args:
            u (dtype_u): current values
            t (float): current time (not used here)

        Returns:
            implicit part of RHS
        """

        tmp = self.dtype_u(self.V)
        tmp.values.vector()[:] = df.assemble(-self.params.nu * df.inner(df.grad(u.values), df.grad(self.v)) * df.dx)
        fimpl = self.__invert_mass_matrix(tmp)

        return fimpl

    def eval_f(self, u, t):
        """
        Routine to evaluate both parts of the RHS

        Args:
            u (dtype_u): current values
            t (float): current time

        Returns:
            dtype_f: the RHS divided into two parts
        """

        f = self.dtype_f(self.V)
        f.impl = self.__eval_fimpl(u, t)
        f.expl = self.__eval_fexpl(u, t)
        return f

    def u_exact(self, t):
        """
        Routine to compute the exact solution at time t

        Args:
            t (float): current time

        Returns:
            dtype_u: exact solution
        """

        u0 = df.Expression('sin(a*x[0]) * cos(t)', a=np.pi, t=t, degree=self.params.order)
        me = self.dtype_u(df.interpolate(u0, self.V))

        return me
