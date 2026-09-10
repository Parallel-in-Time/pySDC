import logging

import dolfin as df
import numpy as np

from pySDC.core.problem import Problem
from pySDC.implementations.datatype_classes.fenics_mesh import fenics_mesh
from pySDC.projects.StroemungsRaum.problem_classes.newton_step import NewtonStep


class _PeriodicX(df.SubDomain):
    """
    Identifies the right boundary :math:`x = 0.5` with the left boundary :math:`x = -0.5`.

    Passed to ``FunctionSpace`` as ``constrained_domain``, so periodicity is handled by
    the dof map itself and the periodic dofs never enter the linear system.
    """

    def inside(self, x, on_boundary):
        return bool(df.near(x[0], -0.5) and on_boundary)

    def map(self, x, y):
        y[0] = x[0] - 1.0
        y[1] = x[1]


class fenics_NSE_2D_TaylorGreen(Problem):
    r"""
    Forced two-dimensional incompressible Navier-Stokes equations on :math:`\Omega = [-0.5, 0.5]^2`,
    set up to expose the order reduction caused by time-dependent Dirichlet boundary conditions.

    .. math::
        \frac{\partial u}{\partial t} = - u \cdot \nabla u + \nu \Delta u - \nabla p + g,
        \qquad \nabla \cdot u = 0

    The forcing :math:`g` is manufactured from the analytical solution

    .. math::
        u(x, y, t) &= 1 - e^{-8\pi^2\nu t}\sin(2\pi(x - t))\sin(\pi y)\cos(\pi y) \\
        v(x, y, t) &= - e^{-8\pi^2\nu t}\cos(2\pi(x - t))\cos^2(\pi y) \\
        p(x, y, t) &= 1 + \frac{4}{17}e^{-16\pi^2\nu t}\cos(4\pi(x - t))\cos(\pi y)

    which is divergence free and exactly one-periodic in :math:`x`. On :math:`y = \pm 0.5` it
    collapses to the constants :math:`u = (1, 0)`, :math:`p = 1`, so the top and bottom boundary
    data is time-independent.

    Because the solution is genuinely periodic in :math:`x`, the *same* exact solution satisfies
    both variants selected by ``periodic``:

    - ``periodic=False``: time-dependent Dirichlet conditions on :math:`x = \pm 0.5`,
    - ``periodic=True``: periodic conditions on :math:`x = \pm 0.5`.

    The only difference between the two runs is therefore the presence of time-dependent boundary
    data, which is what isolates the order reduction.

    The problem is discretized in space with Taylor-Hood elements on a mixed velocity-pressure
    space and solved monolithically, so the semi-discrete system is the differential-algebraic
    system :math:`M \dot{w} = f(w, t)` with the singular mass matrix :math:`M = \mathrm{diag}(M_v, 0)`.
    It therefore requires ``generic_implicit_mass`` as sweeper, which applies :math:`M` where needed
    instead of inverting it.

    Parameters
    ----------
    nelems : int, optional
        Number of elements per spatial direction.
    t0 : float, optional
        Starting time.
    order : int, optional
        Polynomial degree of the velocity space; the pressure space uses ``order - 1``.
    nu : float, optional
        Kinematic viscosity :math:`\nu`.
    periodic : bool, optional
        Use periodic instead of time-dependent Dirichlet conditions on :math:`x = \pm 0.5`.
    Sol_tol : float, optional
        Absolute tolerance for the Newton solver.

    Attributes
    ----------
    V : FunctionSpace
        Velocity space.
    Q : FunctionSpace
        Pressure space.
    W : FunctionSpace
        Mixed velocity-pressure space.
    M : Matrix
        The velocity mass matrix :math:`\mathrm{diag}(M_v, 0)` on the mixed space.
    g : Expression
        Manufactured forcing term.
    bc : list of DirichletBC
        Dirichlet boundary conditions, time-dependent unless ``periodic``.
    bc_hom : list of DirichletBC
        Homogeneous conditions on the Dirichlet part of the boundary only, used to fix the residual.
    fix_bc_for_residual : bool
        Flag indicating that the residual requires special treatment due to boundary conditions.

    References
    ----------
    .. [1] The FEniCS Project Version 1.5. M. S. Alnaes, J. Blechta, J. Hake, A. Johansson, B. Kehlet, A. Logg,
        C. Richardson, J. Ring, M. E. Rognes, G. N. Wells. Archive of Numerical Software (2015).
    """

    dtype_u = fenics_mesh
    dtype_f = fenics_mesh

    df.set_log_active(False)

    def __init__(self, nelems=32, t0=0.0, order=2, nu=0.02, periodic=False, Sol_tol=1e-10):

        # set logger level for FFC and dolfin
        logging.getLogger('FFC').setLevel(logging.WARNING)
        logging.getLogger('UFL').setLevel(logging.WARNING)

        # set solver and form parameters
        df.parameters["form_compiler"]["optimize"] = True
        df.parameters["form_compiler"]["cpp_optimize"] = True

        mesh = df.RectangleMesh(df.Point(-0.5, -0.5), df.Point(0.5, 0.5), nelems, nelems)

        # define function spaces (Taylor-Hood); periodicity is baked into the dof map
        P2 = df.VectorElement("P", mesh.ufl_cell(), order)
        P1 = df.FiniteElement("P", mesh.ufl_cell(), order - 1)
        constraint = _PeriodicX() if periodic else None
        self.W = df.FunctionSpace(mesh, df.MixedElement([P2, P1]), constrained_domain=constraint)
        self.V = df.FunctionSpace(mesh, P2, constrained_domain=constraint)
        self.Q = df.FunctionSpace(mesh, P1, constrained_domain=constraint)

        super().__init__(self.W)
        self._makeAttributeAndRegister(
            'nelems', 't0', 'order', 'nu', 'periodic', 'Sol_tol', localVars=locals(), readOnly=True
        )

        self.logger.debug('DoFs on this level: %d', self.W.dim())

        # trial and test functions on the mixed space
        self.u, self.p = df.TrialFunctions(self.W)
        self.v, self.q = df.TestFunctions(self.W)

        # velocity mass matrix on the mixed space, i.e. diag(M_v, 0)
        self.M = df.assemble(df.inner(self.u, self.v) * df.dx)

        # manufactured solution and the forcing term derived from it
        self.u_ex = df.Expression(
            (
                '1.0 - exp(-8*pi*pi*nu*t)*sin(2*pi*(x[0] - t))*sin(pi*x[1])*cos(pi*x[1])',
                '-exp(-8*pi*pi*nu*t)*cos(2*pi*(x[0] - t))*cos(pi*x[1])*cos(pi*x[1])',
            ),
            pi=np.pi,
            nu=nu,
            t=t0,
            degree=order + 2,
        )
        self.p_ex = df.Expression(
            '1.0 + (4.0/17.0)*exp(-16*pi*pi*nu*t)*cos(4*pi*(x[0] - t))*cos(pi*x[1])',
            pi=np.pi,
            nu=nu,
            t=t0,
            degree=order + 2,
        )
        self.g = df.Expression(
            (
                'pi/34.0*exp(-16*pi*pi*nu*t)*sin(4*pi*(t - x[0]))*cos(pi*x[1])*(32.0 - 17.0*cos(pi*x[1]))',
                '2*pi*pi*nu*exp(-8*pi*pi*nu*t)*cos(2*pi*(t - x[0]))'
                ' - pi*exp(-16*pi*pi*nu*t)*sin(pi*x[1])'
                '*(2.0*pow(cos(pi*x[1]), 3) + 4.0/17.0*cos(4*pi*(t - x[0])))',
            ),
            pi=np.pi,
            nu=nu,
            t=t0,
            degree=order + 2,
        )

        # on y = +-0.5 the exact solution is constant in space and time
        top_bottom = 'near(x[1], -0.5) || near(x[1], 0.5)'
        left_right = 'near(x[0], -0.5) || near(x[0], 0.5)'
        self.bc = [
            df.DirichletBC(self.W.sub(0), df.Constant((1.0, 0.0)), top_bottom),
            df.DirichletBC(self.W.sub(1), df.Constant(1.0), top_bottom),
        ]
        if not periodic:
            self.bc += [
                df.DirichletBC(self.W.sub(0), self.u_ex, left_right),
                df.DirichletBC(self.W.sub(1), self.p_ex, left_right),
            ]

        # the residual is meaningless where the solution is prescribed, but only there: with
        # periodicity the dofs on x = +-0.5 are unknowns and their residual has to be kept
        dirichlet = top_bottom if periodic else 'on_boundary'
        self.bc_hom = [
            df.DirichletBC(self.W.sub(0), df.Constant((0.0, 0.0)), dirichlet),
            df.DirichletBC(self.W.sub(1), df.Constant(0.0), dirichlet),
        ]
        self.fix_bc_for_residual = True

        # residual form for a single node-to-node step, assembled once; `factor` and the
        # boundary/forcing expressions carry the time dependence
        self.factor = df.Constant(0.0)
        self.w = df.Function(self.W)
        u, p = df.split(self.w)

        F = df.dot(u, self.v) * df.dx
        F += self.factor * df.dot(df.dot(u, df.nabla_grad(u)), self.v) * df.dx
        F += self.factor * self.nu * df.inner(df.nabla_grad(u), df.nabla_grad(self.v)) * df.dx
        F -= self.factor * df.dot(p, df.div(self.v)) * df.dx
        F -= self.factor * df.dot(self.g, self.v) * df.dx
        F -= self.factor * df.dot(df.div(u), self.q) * df.dx

        self.step = NewtonStep(F, df.derivative(F, self.w))
        self.newton = df.NewtonSolver()
        self.newton.parameters['absolute_tolerance'] = Sol_tol
        self.newton.parameters['relative_tolerance'] = Sol_tol
        self.newton.parameters['maximum_iterations'] = 20

    def solve_system(self, rhs, factor, u0, t):
        r"""
        Newton solver for :math:`M w + factor \cdot N(w, t) = rhs`, where :math:`N` collects the
        convective, viscous, pressure and divergence terms and ``rhs`` is the mass-weighted
        right-hand side assembled by the sweeper.

        Parameters
        ----------
        rhs : dtype_f
            Right-hand side for the nonlinear system.
        factor : float
            Abbrev. for the node-to-node stepsize (or any other factor required).
        u0 : dtype_u
            Initial guess for the iterative solver.
        t : float
            Current time.

        Returns
        -------
        w : dtype_u
            Solution.
        """
        self.factor.assign(factor)
        self.u_ex.t = t
        self.p_ex.t = t
        self.g.t = t

        self.w.vector()[:] = u0.values.vector()[:]
        self.step.rhs = rhs.values.vector()
        self.step.bcs = self.bc
        self.newton.solve(self.step, self.w.vector())

        me = self.dtype_u(self.W)
        me.values.vector()[:] = self.w.vector()[:]
        return me

    def eval_f(self, w, t):
        r"""
        Routine to evaluate the right-hand side of the problem in weak form, i.e. *without*
        applying :math:`M^{-1}`.

        Parameters
        ----------
        w : dtype_u
            Current values of the numerical solution.
        t : float
            Current time at which the numerical solution is computed.

        Returns
        -------
        f : dtype_f
            The right-hand side.
        """
        u, p = df.split(w.values)
        self.g.t = t

        F = -df.dot(df.dot(u, df.nabla_grad(u)), self.v) * df.dx
        F -= self.nu * df.inner(df.nabla_grad(u), df.nabla_grad(self.v)) * df.dx
        F += df.dot(p, df.div(self.v)) * df.dx
        F += df.dot(self.g, self.v) * df.dx
        F += df.dot(df.div(u), self.q) * df.dx

        f = self.dtype_f(self.W)
        df.assemble(F, tensor=f.values.vector())
        return f

    def apply_mass_matrix(self, w):
        r"""
        Routine to apply the velocity mass matrix.

        Parameters
        ----------
        w : dtype_u
            Current values of the numerical solution.

        Returns
        -------
        me : dtype_u
            The product :math:`M \vec{w}`.
        """
        me = self.dtype_u(self.W)
        self.M.mult(w.values.vector(), me.values.vector())
        return me

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
        self.u_ex.t = t
        self.p_ex.t = t

        me = self.dtype_u(self.W)
        df.assign(me.values.sub(0), df.interpolate(self.u_ex, self.V))
        df.assign(me.values.sub(1), df.interpolate(self.p_ex, self.Q))
        return me

    def fix_residual(self, res):
        """
        Applies homogeneous Dirichlet boundary conditions to the residual, on the Dirichlet part
        of the boundary only.

        Parameters
        ----------
        res : dtype_u
            Residual.
        """
        for bc in self.bc_hom:
            bc.apply(res.values.vector())
        return None
