import logging
import random

import dolfin as df
import numpy as np

from pySDC.core.Errors import ParameterError
from pySDC.core.Problem import ptype
from pySDC.implementations.datatype_classes.fenics_mesh import fenics_mesh


# noinspection PyUnusedLocal
class fenics_grayscott(ptype):
    r"""
    The Gray-Scott system [1]_ describes a reaction-diffusion process of two substances :math:`u` and :math:`v`,
    where they diffuse over time. During the reaction :math:`u` is used up with overall decay rate :math:`B`,
    whereas :math:`v` is produced with feed rate :math:`A`. :math:`D_u,\, D_v` are the diffusion rates for
    :math:`u,\, v`. This process is describes by the one-dimensional model using Dirichlet boundary conditions

    .. math::
        \frac{d u}{d t} = D_u \Delta u - u v^2 + A (1 - u),

    .. math::
        \frac{d v}{d t} = D_v \Delta v + u v^2 - B u

    for :math:`x \in \Omega:[0, 100]`. The *weak formulation* of the problem can be obtained by multiplying the
    system with a test function :math:`q`:

    .. math::
        \int_\Omega u_t q dx = \int_\Omega D_u \Delta u q - u v^2 q + A (1 - u) q\,dx,

    .. math::
        \int_\Omega v_t q dx = \int_\Omega D_v \Delta v q + u v^2 q - B u q\,dx,

    The spatial solve of the weak formulation is realized by FEniCS [2]_.

    Parameters
    ----------
    c_nvars : int, optional
        Spatial resolution, i.e., number of degrees of freedom in space.
    t0 : float, optional
        Starting time.
    family : str, optional
        Indicates the family of elements used to create the function space
        for the trail and test functions. The default is 'CG', which are the class
        of Continuous Galerkin, a *synonym* for the Lagrange family of elements, see [3]_.
    order : int, optional
        Defines the order of the elements in the function space.
    refinements : list or tuple, optional
        Defines the refinement for the spatial grid. Needs to be a list or tuple, e.g.
        [2, 2] or (2, 2).
    Du : float, optional
        Diffusion rate for :math:`u`.
    Dv: float, optional
        Diffusion rate for :math:`v`.
    A : float, optional
        Feed rate for :math:`v`.
    B : float, optional
        Overall decay rate for :math:`u`.

    Attributes
    ----------
    V : FunctionSpace
        Defines the function space of the trial and test functions.
    w : Function
        Function for the right-hand side.
    w1 : Function
        Split of w, part 1.
    w2 : Function
        Split of w, part 2.
    F1 : scalar, vector, matrix or higher rank tensor
        Weak form of right-hand side, first part.
    F2 : scalar, vector, matrix or higher rank tensor
        Weak form of right-hand side, second part.
    F : scalar, vector, matrix or higher rank tensor
        Weak form of full right-hand side.
    M : matrix
        Full mass matrix for both parts.

    References
    ----------
    .. [1] Autocatalytic reactions in the isothermal, continuous stirred tank reactor: Isolas and other forms
        of multistability. P. Gray, S. K. Scott. Chem. Eng. Sci. 38, 1 (1983).
    .. [2] The FEniCS Project Version 1.5. M. S. Alnaes, J. Blechta, J. Hake, A. Johansson, B. Kehlet, A. Logg,
        C. Richardson, J. Ring, M. E. Rognes, G. N. Wells. Archive of Numerical Software (2015).
    .. [3] Automated Solution of Differential Equations by the Finite Element Method. A. Logg, K.-A. Mardal, G. N.
        Wells and others. Springer (2012).
    """

    dtype_u = fenics_mesh
    dtype_f = fenics_mesh

    def __init__(self, c_nvars=256, t0=0.0, family='CG', order=4, refinements=None, Du=1.0, Dv=0.01, A=0.09, B=0.086):
        """Initialization routine"""

        if refinements is None:
            refinements = [1, 0]

        # define the Dirichlet boundary
        def Boundary(x, on_boundary):
            return on_boundary

        # set logger level for FFC and dolfin
        df.set_log_level(df.WARNING)
        logging.getLogger('FFC').setLevel(logging.WARNING)

        # set solver and form parameters
        df.parameters["form_compiler"]["optimize"] = True
        df.parameters["form_compiler"]["cpp_optimize"] = True

        # set mesh and refinement (for multilevel)
        mesh = df.IntervalMesh(c_nvars, 0, 100)
        for _ in range(refinements):
            mesh = df.refine(mesh)

        # define function space for future reference
        V = df.FunctionSpace(mesh, family, order)
        self.V = V * V

        # invoke super init, passing number of dofs
        super(fenics_grayscott).__init__(V)
        self._makeAttributeAndRegister(
            'c_nvars', 't0', 'family', 'order', 'refinements', 'Du', 'Dv', 'A', 'B', localVars=locals(), readOnly=True
        )
        # rhs in weak form
        self.w = df.Function(self.V)
        q1, q2 = df.TestFunctions(self.V)

        self.w1, self.w2 = df.split(self.w)

        self.F1 = (
            -self.Du * df.inner(df.nabla_grad(self.w1), df.nabla_grad(q1))
            - self.w1 * (self.w2**2) * q1
            + self.A * (1 - self.w1) * q1
        ) * df.dx
        self.F2 = (
            -self.Dv * df.inner(df.nabla_grad(self.w2), df.nabla_grad(q2))
            + self.w1 * (self.w2**2) * q2
            - self.B * self.w2 * q2
        ) * df.dx
        self.F = self.F1 + self.F2

        # mass matrix
        u1, u2 = df.TrialFunctions(self.V)
        a_M = u1 * q1 * df.dx
        M1 = df.assemble(a_M)
        a_M = u2 * q2 * df.dx
        M2 = df.assemble(a_M)
        self.M = M1 + M2

    def __invert_mass_matrix(self, u):
        r"""
        Helper routine to invert mass matrix.

        Parameters
        ----------
        u : dtype_u
            Current values of the numerical solution.

        Returns
        -------
        me : dtype_u
            The product :math:`M^{-1} \vec{u}`.
        """

        me = self.dtype_u(self.V)

        A = 1.0 * self.M
        b = self.dtype_u(u)

        df.solve(A, me.values.vector(), b.values.vector())

        return me

    def solve_system(self, rhs, factor, u0, t):
        r"""
        Dolfin's linear solver for :math:`(M - factor A) \vec{u} = \vec{rhs}`.

        Parameters
        ----------
        rhs : dtype_f
            Right-hand side for the nonlinear system.
        factor : float
            Abbrev. for the node-to-node stepsize (or any other factor required).
        u0 : dtype_u
            Initial guess for the iterative solver (not used here so far).
        t : float
            Current time.

        Returns
        -------
        me : dtype_u
            Solution as mesh.
        """

        sol = self.dtype_u(self.V)

        self.w.assign(sol.values)

        # fixme: is this really necessary to do each time?
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
        prm['newton_solver']['absolute_tolerance'] = 1e-09
        prm['newton_solver']['relative_tolerance'] = 1e-08
        prm['newton_solver']['maximum_iterations'] = 100
        prm['newton_solver']['relaxation_parameter'] = 1.0

        solver.solve()

        sol.values.assign(self.w)

        return sol

    def eval_f(self, u, t):
        """
        Routine to evaluate both parts of the right-hand side.

        Parameters
        ----------
        u : dtype_u
            Current values of the numerical solution.
        t : float
            Current time at which the numerical solution is computed.

        Returns
        -------
        f : dtype_f
            The right-hand side divided into two parts.
        """

        f = self.dtype_f(self.V)

        self.w.assign(u.values)
        f.values = df.Function(self.V, df.assemble(self.F))

        f = self.__invert_mass_matrix(f)

        return f

    def u_exact(self, t):
        r"""
        Routine to compute the exact solution at time t.

        Parameters
        ----------
        t : float
            Time of the exact solution.

        Returns
        -------
        me : dtype_u
            Exact solution (only at :math:`t_0 = 0.0`).
        """

        class InitialConditions(df.Expression):
            def __init__(self):
                # fixme: why do we need this?
                random.seed(2)
                pass

            def eval(self, values, x):
                values[0] = 1 - 0.5 * np.power(np.sin(np.pi * x[0] / 100), 100)
                values[1] = 0.25 * np.power(np.sin(np.pi * x[0] / 100), 100)

            def value_shape(self):
                return (2,)

        assert t == 0, 'ERROR: u_exact only valid for t=0'

        uinit = InitialConditions()

        me = self.dtype_u(self.V)
        me.values = df.interpolate(uinit, self.V)

        return me
