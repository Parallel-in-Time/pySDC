
import logging

import dolfin as df
import numpy as np

from pySDC.core.Errors import ParameterError
from pySDC.core.Problem import ptype
from pySDC.implementations.datatype_classes.fenics_mesh import fenics_mesh, rhs_fenics_mesh


# noinspection PyUnusedLocal
class fenics_vortex_2d(ptype):
    """
    Example implementing the 2d vorticity-velocity equation with periodic BC in [0,1]

    Attributes:
        V: function space
        M: mass matrix for FEM
        K: stiffness matrix incl. diffusion coefficient (and correct sign)
    """

    def __init__(self, problem_params, dtype_u=fenics_mesh, dtype_f=rhs_fenics_mesh):
        """
        Initialization routine

        Args:
            problem_params (dict): custom parameters for the example
            dtype_u: FEniCS mesh data type (will be passed to parent class)
            dtype_f: FEniCS mesh data data type with implicit and explicit parts (will be passed to parent class)
        """

        # Sub domain for Periodic boundary condition
        class PeriodicBoundary(df.SubDomain):

            # Left boundary is "target domain" G
            def inside(self, x, on_boundary):
                # return True if on left or bottom boundary AND NOT on one of the two corners (0, 1) and (1, 0)
                return bool((df.near(x[0], 0) or df.near(x[1], 0)) and
                            (not ((df.near(x[0], 0) and df.near(x[1], 1)) or
                                  (df.near(x[0], 1) and df.near(x[1], 0)))) and on_boundary)

            def map(self, x, y):
                if df.near(x[0], 1) and df.near(x[1], 1):
                    y[0] = x[0] - 1.
                    y[1] = x[1] - 1.
                elif df.near(x[0], 1):
                    y[0] = x[0] - 1.
                    y[1] = x[1]
                else:  # near(x[1], 1)
                    y[0] = x[0]
                    y[1] = x[1] - 1.

        # these parameters will be used later, so assert their existence
        essential_keys = ['c_nvars', 'family', 'order', 'refinements', 'nu', 'rho', 'delta']
        for key in essential_keys:
            if key not in problem_params:
                msg = 'need %s to instantiate problem, only got %s' % (key, str(problem_params.keys()))
                raise ParameterError(msg)

        # set logger level for FFC and dolfin
        df.set_log_level(df.WARNING)
        logging.getLogger('FFC').setLevel(logging.WARNING)

        # set solver and form parameters
        df.parameters["form_compiler"]["optimize"] = True
        df.parameters["form_compiler"]["cpp_optimize"] = True

        # set mesh and refinement (for multilevel)
        mesh = df.UnitSquareMesh(problem_params['c_nvars'][0], problem_params['c_nvars'][1])
        for i in range(problem_params['refinements']):
            mesh = df.refine(mesh)

        self.mesh = df.Mesh(mesh)

        # define function space for future reference
        self.V = df.FunctionSpace(mesh, problem_params['family'], problem_params['order'],
                                  constrained_domain=PeriodicBoundary())
        tmp = df.Function(self.V)
        print('DoFs on this level:', len(tmp.vector().vector()[:]))

        # invoke super init, passing number of dofs, dtype_u and dtype_f
        super(fenics_vortex_2d, self).__init__(self.V, dtype_u, dtype_f, problem_params)

        w = df.TrialFunction(self.V)
        v = df.TestFunction(self.V)

        # Stiffness term (diffusion)
        a_K = df.inner(df.nabla_grad(w), df.nabla_grad(v)) * df.dx

        # Mass term
        a_M = w * v * df.dx

        self.M = df.assemble(a_M)
        self.K = df.assemble(a_K)

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

        A = self.M + self.nu * factor * self.K
        b = self.__apply_mass_matrix(rhs)

        u = self.dtype_u(u0)
        df.solve(A, u.values.vector(), b.values.vector())

        return u

    def __eval_fexpl(self, u, t):
        """
        Helper routine to evaluate the explicit part of the RHS

        Args:
            u (dtype_u): current values
            t (float): current time

        Returns:
            explicit part of RHS
        """

        A = 1.0 * self.K
        b = self.__apply_mass_matrix(u)
        psi = self.dtype_u(self.V)
        df.solve(A, psi.values.vector(), b.values.vector())

        fexpl = self.dtype_u(self.V)
        fexpl.values = df.project(df.Dx(psi.values, 1) * df.Dx(u.values, 0) - df.Dx(psi.values, 0) * df.Dx(u.values, 1),
                                  self.V)

        return fexpl

    def __eval_fimpl(self, u, t):
        """
        Helper routine to evaluate the implicit part of the RHS

        Args:
            u (dtype_u): current values
            t (float): current time

        Returns:
            implicit part of RHS
        """

        tmp = self.dtype_u(self.V)
        tmp.values = df.Function(self.V, -1.0 * self.params.nu * self.K * u.values.vector())
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

    def __apply_mass_matrix(self, u):
        """
        Routine to apply mass matrix

        Args:
            u (dtype_u): current values

        Returns:
            dtype_u: M*u
        """

        me = self.dtype_u(self.V)
        me.values = df.Function(self.V, self.M * u.values.vector())

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

        A = 1.0 * self.M
        b = self.dtype_u(u)

        df.solve(A, me.values.vector(), b.values.vector())

        return me

    def u_exact(self, t):
        """
        Routine to compute the exact solution at time t

        Args:
            t (float): current time

        Returns:
            dtype_u: exact solution
        """

        w = df.Expression('r*(1-pow(tanh(r*((0.75-4) - x[1])),2)) + r*(1-pow(tanh(r*(x[1] - (0.25-4))),2)) - \
                           r*(1-pow(tanh(r*((0.75-3) - x[1])),2)) + r*(1-pow(tanh(r*(x[1] - (0.25-3))),2)) - \
                           r*(1-pow(tanh(r*((0.75-2) - x[1])),2)) + r*(1-pow(tanh(r*(x[1] - (0.25-2))),2)) - \
                           r*(1-pow(tanh(r*((0.75-1) - x[1])),2)) + r*(1-pow(tanh(r*(x[1] - (0.25-1))),2)) - \
                           r*(1-pow(tanh(r*((0.75-0) - x[1])),2)) + r*(1-pow(tanh(r*(x[1] - (0.25-0))),2)) - \
                           r*(1-pow(tanh(r*((0.75+1) - x[1])),2)) + r*(1-pow(tanh(r*(x[1] - (0.25+1))),2)) - \
                           r*(1-pow(tanh(r*((0.75+2) - x[1])),2)) + r*(1-pow(tanh(r*(x[1] - (0.25+2))),2)) - \
                           r*(1-pow(tanh(r*((0.75+3) - x[1])),2)) + r*(1-pow(tanh(r*(x[1] - (0.25+3))),2)) - \
                           r*(1-pow(tanh(r*((0.75+4) - x[1])),2)) + r*(1-pow(tanh(r*(x[1] - (0.25+4))),2)) - \
                           d*2*a*cos(2*a*(x[0]+0.25))',
                          d=self.params.delta, r=self.params.rho, a=np.pi, degree=self.params.order)

        me = self.dtype_u(self.V)
        me.values = df.interpolate(w, self.V)

        # df.plot(me.values)
        # df.interactive()
        # exit()

        return me
