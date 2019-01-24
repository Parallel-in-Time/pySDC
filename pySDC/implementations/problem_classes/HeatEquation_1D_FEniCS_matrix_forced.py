
import dolfin as df
import numpy as np
import logging

from pySDC.implementations.datatype_classes.fenics_mesh import fenics_mesh, rhs_fenics_mesh

from pySDC.core.Problem import ptype
from pySDC.core.Errors import ParameterError


# noinspection PyUnusedLocal
class fenics_heat(ptype):
    """
    Example implementing the forced 1D heat equation with Dirichlet-0 BC in [0,1]

    Attributes:
        V: function space
        M: mass matrix for FEM
        K: stiffness matrix incl. diffusion coefficient (and correct sign)
        g: forcing term
        bc: boundary conditions
    """

    def __init__(self, problem_params, dtype_u=fenics_mesh, dtype_f=rhs_fenics_mesh):
        """
        Initialization routine

        Args:
            problem_params (dict): custom parameters for the example
            dtype_u: FEniCS mesh data type (will be passed to parent class)
            dtype_f: FEniCS mesh data data type with implicit and explicit parts (will be passed to parent class)
        """

        # define the Dirichlet boundary
        # def Boundary(x, on_boundary):
        #     return on_boundary

        # these parameters will be used later, so assert their existence
        essential_keys = ['c_nvars', 't0', 'family', 'order', 'refinements', 'nu']
        for key in essential_keys:
            if key not in problem_params:
                msg = 'need %s to instantiate problem, only got %s' % (key, str(problem_params.keys()))
                raise ParameterError(msg)

        # set logger level for FFC and dolfin
        logging.getLogger('FFC').setLevel(logging.WARNING)
        logging.getLogger('UFL').setLevel(logging.WARNING)

        # set solver and form parameters
        df.parameters["form_compiler"]["optimize"] = True
        df.parameters["form_compiler"]["cpp_optimize"] = True
        df.parameters['allow_extrapolation'] = True

        # set mesh and refinement (for multilevel)
        mesh = df.UnitIntervalMesh(problem_params['c_nvars'])
        for i in range(problem_params['refinements']):
            mesh = df.refine(mesh)

        # define function space for future reference
        self.V = df.FunctionSpace(mesh, problem_params['family'], problem_params['order'])
        tmp = df.Function(self.V)
        print('DoFs on this level:', len(tmp.vector()[:]))

        # invoke super init, passing number of dofs, dtype_u and dtype_f
        super(fenics_heat, self).__init__(self.V, dtype_u, dtype_f, problem_params)

        # Stiffness term (Laplace)
        u = df.TrialFunction(self.V)
        v = df.TestFunction(self.V)
        a_K = -1.0 * df.inner(df.nabla_grad(u), self.params.nu * df.nabla_grad(v)) * df.dx

        # Mass term
        a_M = u * v * df.dx

        self.M = df.assemble(a_M)
        self.K = df.assemble(a_K)

        # set forcing term as expression
        self.g = df.Expression('-cos(a*x[0]) * (sin(t) - b*a*a*cos(t))', a=np.pi, b=self.params.nu, t=self.params.t0,
                               degree=self.params.order)
        # self.g = df.Expression('0', a=np.pi, b=self.params.nu, t=self.params.t0,
        #                        degree=self.params.order)
        # set boundary values
        # bc = df.DirichletBC(self.V, df.Constant(0.0), Boundary)
        #
        # bc.apply(self.M)
        # bc.apply(self.K)

    def solve_system(self, rhs, factor, u0, t):
        """
        Dolfin's linear solver for (M-dtA)u = rhs

        Args:
            rhs (dtype_f): right-hand side for the nonlinear system
            factor (float): abbrev. for the node-to-node stepsize (or any other factor required)
            u0 (dtype_u_: initial guess for the iterative solver (not used here so far)
            t (float): current time

        Returns:
            dtype_u: solution as mesh
        """

        b = self.apply_mass_matrix(rhs)

        u = self.dtype_u(u0)
        df.solve(self.M - factor * self.K, u.values.vector(), b.values.vector())

        return u

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
        self.K.mult(u.values.vector(), tmp.values.vector())
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

    def apply_mass_matrix(self, u):
        """
        Routine to apply mass matrix

        Args:
            u (dtype_u): current values

        Returns:
            dtype_u: M*u
        """

        me = self.dtype_u(self.V)
        self.M.mult(u.values.vector(), me.values.vector())

        return me

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

        df.solve(self.M, me.values.vector(), b.values.vector())

        return me

    def u_exact(self, t):
        """
        Routine to compute the exact solution at time t

        Args:
            t (float): current time

        Returns:
            dtype_u: exact solution
        """

        u0 = df.Expression('cos(a*x[0]) * cos(t)', a=np.pi, t=t, degree=self.params.order)
        me = self.dtype_u(df.interpolate(u0, self.V))

        return me


# noinspection PyUnusedLocal
class fenics_heat_mass(fenics_heat):
    """
    Example implementing the forced 1D heat equation with Dirichlet-0 BC in [0,1], expects mass matrix sweeper

    """

    def solve_system(self, rhs, factor, u0, t):
        """
        Dolfin's linear solver for (M-dtA)u = rhs

        Args:
            rhs (dtype_f): right-hand side for the nonlinear system
            factor (float): abbrev. for the node-to-node stepsize (or any other factor required)
            u0 (dtype_u_: initial guess for the iterative solver (not used here so far)
            t (float): current time

        Returns:
            dtype_u: solution as mesh
        """

        u = self.dtype_u(u0)
        df.solve(self.M - factor * self.K, u.values.vector(), rhs.values.vector())

        return u

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

        self.K.mult(u.values.vector(), f.impl.values.vector())

        self.g.t = t
        f.expl = self.dtype_u(df.interpolate(self.g, self.V))
        f.expl = self.apply_mass_matrix(f.expl)

        return f
