import logging

import dolfin as df
import numpy as np

from pySDC.core.Problem import ptype
from pySDC.implementations.datatype_classes.fenics_mesh import fenics_mesh, rhs_fenics_mesh


# noinspection PyUnusedLocal
class fenics_heat_weak_fullyimplicit(ptype):
    r"""
    Example implementing the forced one-dimensional heat equation with Dirichlet boundary conditions

    .. math::
        \frac{d u}{d t} = \nu \frac{d^2 u}{d x^2} + f

    for :math:`x \in \Omega:=[0,1]`, where the forcing term :math:`f` is defined by

    .. math::
        f(x, t) = -\sin(\pi x) (\sin(t) - \nu \pi^2 \cos(t)).

    The exact solution of the problem is

    .. math::
        u(x, t) = \sin(\pi x)\cos(t).

    In this class the problem is implemented in the way that the spatial part is solved using ``FEniCS`` [1]_. Hence, the problem
    is reformulated to the *weak formulation*

    .. math:
        \int_\Omega u_t v\,dx = - \nu \int_\Omega \nabla u \nabla v\,dx + \int_\Omega f v\,dx.

    The nonlinear system is solved in a *fully-implicit* way using Dolfin's weak solver provided by the routine
    ``df.NonlinearVariationalSolver``.

    Parameters
    ----------
    c_nvars : int, optional
        Spatial resolution, i.e., numbers of degrees of freedom in space.
    t0 : float, optional
        Starting time.
    family : str, optional
        Indicates the family of elements used to create the function space
        for the trail and test functions. The default is ``'CG'``, which are the class
        of Continuous Galerkin, a *synonym* for the Lagrange family of elements, see [2]_.
    order : int, optional
        Defines the order of the elements in the function space.
    refinements : int, optional
        Denotes the refinement of the mesh. ``refinements=2`` refines the mesh by factor :math:`2`.
    nu : float, optional
        Diffusion coefficient :math:`\nu`.

    Attributes
    ----------
    V : FunctionSpace
        Defines the function space of the trial and test functions.
    w : Function
        Function for the weak form.
    a_K : scalar, vector, matrix or higher rank tensor
        The expression :math:`- \nu \int_\Omega \nabla u \nabla v\,dx + \int_\Omega f v\,dx` (incl. BC).
    M : scalar, vector, matrix or higher rank tensor
        Denotes the expression :math:`\int_\Omega u_t v\,dx`.
    g : Expression
        The forcing term :math:`f` in the heat equation.
    bc : DirichletBC
        Denotes the Dirichlet boundary conditions.

    References
    ----------
    .. [1] The FEniCS Project Version 1.5. M. S. Alnaes, J. Blechta, J. Hake, A. Johansson, B. Kehlet, A. Logg,
        C. Richardson, J. Ring, M. E. Rognes, G. N. Wells. Archive of Numerical Software (2015).
    .. [2] Automated Solution of Differential Equations by the Finite Element Method. A. Logg, K.-A. Mardal, G. N.
        Wells and others. Springer (2012).
    """

    dtype_u = fenics_mesh
    dtype_f = fenics_mesh

    def __init__(self, c_nvars=128, t0=0.0, family='CG', order=4, refinements=1, nu=0.1):
        """Initialization routine"""

        # define the Dirichlet boundary
        def Boundary(x, on_boundary):
            return on_boundary

        # these parameters will be used later, so assert their existence
        essential_keys = ['c_nvars', 't0', 'family', 'order', 'refinements', 'nu']

        # set logger level for FFC and dolfin
        logging.getLogger('ULF').setLevel(logging.WARNING)
        logging.getLogger('FFC').setLevel(logging.WARNING)
        df.set_log_level(30)

        # set solver and form parameters
        df.parameters["form_compiler"]["optimize"] = True
        df.parameters["form_compiler"]["cpp_optimize"] = True

        # set mesh and refinement (for multilevel)
        mesh = df.UnitIntervalMesh(c_nvars)
        for _ in range(refinements):
            mesh = df.refine(mesh)

        # define function space for future reference
        self.V = df.FunctionSpace(mesh, family, order)
        tmp = df.Function(self.V)
        print('DoFs on this level:', len(tmp.vector()[:]))

        # invoke super init, passing number of dofs, dtype_u and dtype_f
        super(fenics_heat_weak_fullyimplicit, self).__init__(self.V)
        self._makeAttributeAndRegister(
            'c_nvars', 't0', 'family', 'order', 'refinements', 'nu', localVars=locals(), readOnly=True
        )

        self.g = df.Expression(
            '-sin(a*x[0]) * (sin(t) - b*a*a*cos(t))',
            a=np.pi,
            b=self.nu,
            t=self.t0,
            degree=self.order,
        )

        # rhs in weak form
        self.w = df.Function(self.V)
        v = df.TestFunction(self.V)
        self.a_K = -self.nu * df.inner(df.nabla_grad(self.w), df.nabla_grad(v)) * df.dx + self.g * v * df.dx

        # mass matrix
        u = df.TrialFunction(self.V)
        a_M = u * v * df.dx
        self.M = df.assemble(a_M)

        self.bc = df.DirichletBC(self.V, df.Constant(0.0), Boundary)

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

        self.bc.apply(A, b.values.vector())

        df.solve(A, me.values.vector(), b.values.vector())

        return me

    def solve_system(self, rhs, factor, u0, t):
        r"""
        Dolfin's weak solver for :math:`(M - factor \cdot A) \vec{u} = \vec{rhs}`.

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
        sol : dtype_u
            Solution.
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

        self.g.t = t

        f = self.dtype_f(self.V)

        self.w.assign(u.values)
        f.values = df.Function(self.V, df.assemble(self.a_K))

        f = self.__invert_mass_matrix(f)

        return f

    def u_exact(self, t):
        r"""
        Routine to compute the exact solution at time :math:`t`.

        Parameters
        ----------
        t : float
            Time of the exact solution.

        Returns
        -------
        me : dtype_u
            Exact solution.
        """

        u0 = df.Expression('sin(a*x[0]) * cos(t)', a=np.pi, t=t, degree=self.order)
        me = self.dtype_u(self.V)
        me.values = df.interpolate(u0, self.V)

        return me


class fenics_heat_weak_imex(ptype):
    r"""
    Example implementing the forced one-dimensional heat equation with Dirichlet boundary conditions

    .. math::
        \frac{d u}{d t} = \nu \frac{d^2 u}{d x^2} + f

    for :math:`x \in \Omega:=[0,1]`, where the forcing term :math:`f` is defined by

    .. math::
        f(x, t) = -\sin(\pi x) (\sin(t) - \nu \pi^2 \cos(t)).

    The exact solution of the problem is

    .. math::
        u(x, t) = \sin(\pi x)\cos(t).

    In this class the problem is implemented in the way that the spatial part is solved using ``FEniCS`` [1]_. Hence, the problem
    is reformulated to the *weak formulation*

    .. math:
        \int_\Omega u_t v\,dx = - \nu \int_\Omega \nabla u \nabla v\,dx + \int_\Omega f v\,dx.

    The problem is solved in a *semi-explicit* way, i.e., the part containing the forcing term is treated explicitly, where
    it is interpolated in the function space. The first expression in the right-hand side of the weak formulations is solved
    implicitly.

    Parameters
    ----------
    c_nvars : int, optional
        Spatial resolution, i.e., numbers of degrees of freedom in space.
    t0 : float, optional
        Starting time.
    family : str, optional
        Indicates the family of elements used to create the function space
        for the trail and test functions. The default is ``'CG'``, which are the class
        of Continuous Galerkin, a *synonym* for the Lagrange family of elements, see [2]_.
    order : int, optional
        Defines the order of the elements in the function space.
    refinements : int, optional
        Denotes the refinement of the mesh. ``refinements=2`` refines the mesh by factor :math:`2`.
    nu : float, optional
        Diffusion coefficient :math:`\nu`.

    Attributes
    ----------
    V : FunctionSpace
        Defines the function space of the trial and test functions.
    u : TrialFunction
        The unknown function of the problem.
    v : TestFunction
        The test function for the weak form.
    a_K : scalar, vector, matrix or higher rank tensor
        The expression :math:`- \nu \int_\Omega \nabla u \nabla v\,dx + \int_\Omega f v\,dx` (incl. BC).
    M : scalar, vector, matrix or higher rank tensor
        Denotes the expression :math:`\int_\Omega u_t v\,dx`.
    g : Expression
        The forcing term :math:`f` in the heat equation.
    bc : DirichletBC
        Denotes the Dirichlet boundary conditions.

    References
    ----------
    .. [1] The FEniCS Project Version 1.5. M. S. Alnaes, J. Blechta, J. Hake, A. Johansson, B. Kehlet, A. Logg,
        C. Richardson, J. Ring, M. E. Rognes, G. N. Wells. Archive of Numerical Software (2015).
    .. [2] Automated Solution of Differential Equations by the Finite Element Method. A. Logg, K.-A. Mardal, G. N.
        Wells and others. Springer (2012).
    """

    dtype_u = fenics_mesh
    dtype_f = rhs_fenics_mesh

    def __init__(self, c_nvars=128, t0=0.0, family='CG', order=4, refinements=1, nu=0.1):
        """Initialization routine"""

        # define the Dirichlet boundary
        def Boundary(x, on_boundary):
            return on_boundary

        # set logger level for FFC and dolfin
        logging.getLogger('ULF').setLevel(logging.WARNING)
        logging.getLogger('FFC').setLevel(logging.WARNING)
        df.set_log_level(30)

        # set solver and form parameters
        df.parameters["form_compiler"]["optimize"] = True
        df.parameters["form_compiler"]["cpp_optimize"] = True

        # set mesh and refinement (for multilevel)
        mesh = df.UnitIntervalMesh(c_nvars)
        for _ in range(refinements):
            mesh = df.refine(mesh)

        # define function space for future reference
        self.V = df.FunctionSpace(mesh, family, order)
        tmp = df.Function(self.V)
        print('DoFs on this level:', len(tmp.vector()[:]))

        # invoke super init, passing number of dofs, dtype_u and dtype_f
        super(fenics_heat_weak_imex, self).__init__(self.V)
        self._makeAttributeAndRegister(
            'c_nvars', 't0', 'family', 'order', 'refinements', 'nu', localVars=locals(), readOnly=True
        )

        self.g = df.Expression(
            '-sin(a*x[0]) * (sin(t) - b*a*a*cos(t))',
            a=np.pi,
            b=self.nu,
            t=self.t0,
            degree=self.order,
        )

        # rhs in weak form
        self.u = df.TrialFunction(self.V)
        self.v = df.TestFunction(self.V)
        self.a_K = -self.nu * df.inner(df.grad(self.u), df.grad(self.v)) * df.dx

        # mass matrix
        a_M = self.u * self.v * df.dx
        self.M = df.assemble(a_M)

        self.bc = df.DirichletBC(self.V, df.Constant(0.0), Boundary)

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

        b = self.dtype_u(u)

        self.bc.apply(self.M, b.values.vector())

        df.solve(self.M, me.values.vector(), b.values.vector())

        return me

    def solve_system(self, rhs, factor, u0, t):
        r"""
        Dolfin's weak solver for :math:`(M - factor \cdot A)\vec{u} = \vec{u}`.

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
        sol : dtype_u
            Solution.
        """

        sol = self.dtype_u(u0)

        df.solve(self.u * self.v * df.dx - factor * self.a_K == rhs.values * self.v * df.dx, sol.values, self.bc)

        return sol

    def __eval_fexpl(self, u, t):
        """
        Helper routine to evaluate the explicit part of the right-hand side.

        Parameters
        ----------
        u : dtype_u
            Current values of the numerical solution (not used here).
        t : float
            Current time at which the numerical solution is computed.

        Returns
        -------
        fexpl : dtype_u
            Explicit part of the right-hand side.
        """

        self.g.t = t
        fexpl = self.dtype_u(df.interpolate(self.g, self.V))

        return fexpl

    def __eval_fimpl(self, u, t):
        """
        Helper routine to evaluate the implicit part of the right-hand side.

        Parameters
        ----------
        u : dtype_u
            Current values of the numerical solution.
        t : float
            Current time at which the numerical solution is computed (not used here).

        Returns
        -------
        fimpl : dtype_u
            Implicit part of the right-hand side.
        """

        tmp = self.dtype_u(self.V)
        tmp.values.vector()[:] = df.assemble(-self.nu * df.inner(df.grad(u.values), df.grad(self.v)) * df.dx)
        fimpl = self.__invert_mass_matrix(tmp)

        return fimpl

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
        f.impl = self.__eval_fimpl(u, t)
        f.expl = self.__eval_fexpl(u, t)
        return f

    def u_exact(self, t):
        r"""
        Routine to compute the exact solution at time :math:`t`.

        Parameters
        ----------
        t : float
            Time of the exact solution.

        Returns
        -------
        me : dtype_u
            Exact solution.
        """

        u0 = df.Expression('sin(a*x[0]) * cos(t)', a=np.pi, t=t, degree=self.order)
        me = self.dtype_u(df.interpolate(u0, self.V))

        return me
