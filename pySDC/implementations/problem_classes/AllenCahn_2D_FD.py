import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from pySDC.core.errors import ParameterError, ProblemError
from pySDC.core.problem import Problem, WorkCounter
from pySDC.helpers import problem_helper
from pySDC.implementations.datatype_classes.mesh import mesh, imex_mesh, comp2_mesh

# http://www.personal.psu.edu/qud2/Res/Pre/dz09sisc.pdf


# noinspection PyUnusedLocal
class allencahn_fullyimplicit(Problem):
    r"""
    Example implementing the two-dimensional Allen-Cahn equation with periodic boundary conditions, with the two
    phases at :math:`u = 0` and :math:`u = 1`

    .. math::
        \frac{\partial u}{\partial t} = \Delta u
            + \frac{1}{2\varepsilon^2} (2u - 1)\left(1 - (2u - 1)^\nu\right)

    for a constant :math:`\nu`, which at the default :math:`\nu = 2` is the usual
    :math:`\Delta u - \frac{2}{\varepsilon^2} u (1 - u)(1 - 2u)`.

    Initial condition are circles of the form

    .. math::
        u({\bf x}, 0) = \frac{1}{2}\left(1 + \tanh\left(\frac{r - \sqrt{x_i^2 + y_j^2}}
        {\sqrt{2}\varepsilon}\right)\right)

    for :math:`i, j=0,..,N-1`, where :math:`N` is the number of spatial grid points. For time-stepping, the problem is
    treated *fully-implicitly*, i.e., the nonlinear system is solved by Newton.

    Parameters
    ----------
    nvars : tuple of int, optional
        Number of unknowns in the problem, e.g. ``nvars=(128, 128)``.
    nu : int, optional
        Exponent of the double well; :math:`\nu = 2` is the standard Allen-Cahn nonlinearity.
    eps : float, optional
        Scaling parameter :math:`\varepsilon`.
    newton_maxiter : int, optional
        Maximum number of iterations for the Newton solver.
    newton_tol : float, optional
        Tolerance for Newton's method to terminate.
    lin_tol : float, optional
        Tolerance for linear solver to terminate.
    lin_maxiter : int, optional
        Maximum number of iterations for the linear solver.
    radius : float, optional
        Radius of the circles.
    order : int, optional
        Order of the finite difference matrix.
    useGPU : bool, optional
        Run on the GPU with CuPy instead of on the CPU with NumPy.

    Attributes
    ----------
    A : scipy.spdiags
        Second-order FD discretization of the 2D laplace operator.
    dx : float
        Distance between two spatial nodes (same for both directions).
    xvalues : np.1darray
        Spatial grid points, here both dimensions have the same grid points.
    newton_ncalls : int
        Number of calls of the Newton solver. The iterations themselves are counted in
        ``work_counters['newton']``, and the linear ones in ``work_counters['linear']``.
    lin_ncalls : int
        Number of calls of the linear solver.
    """

    dtype_u = mesh
    dtype_f = mesh

    xp = np
    xsp = sp
    linalg = spla

    def setup_GPU(self):
        """
        Switch the array, sparse and solver modules and the datatypes over to CuPy.

        This changes the class, not the instance, as everything else in pySDC that does this
        does: once one instance of a class runs on the GPU, they all do.
        """
        import cupy as cp
        import cupyx.scipy.sparse as csp
        import cupyx.scipy.sparse.linalg as cspla
        from pySDC.implementations.datatype_classes.cupy_mesh import cupy_mesh, imex_cupy_mesh, comp2_cupy_mesh

        self.xp = cp
        self.xsp = csp
        self.linalg = cspla
        self.dtype_u = cupy_mesh
        # .get, not [], because this runs once per instance and the class keeps what it is given
        GPU_versions = {mesh: cupy_mesh, imex_mesh: imex_cupy_mesh, comp2_mesh: comp2_cupy_mesh}
        self.dtype_f = GPU_versions.get(self.dtype_f, self.dtype_f)

    def __init__(
        self,
        nvars=(128, 128),
        nu=2,
        eps=0.04,
        newton_maxiter=200,
        newton_tol=1e-12,
        lin_tol=1e-8,
        lin_maxiter=100,
        inexact_linear_ratio=None,
        radius=0.25,
        order=2,
        useGPU=False,
    ):
        """Initialization routine"""
        if useGPU:
            self.setup_GPU()

        # we assert that nvars looks very particular here.. this will be necessary for coarsening in space later on
        if len(nvars) != 2:
            raise ProblemError('this is a 2d example, got %s' % nvars)
        if nvars[0] != nvars[1]:
            raise ProblemError('need a square domain, got %s' % nvars)
        if nvars[0] % 2 != 0:
            raise ProblemError('the setup requires nvars = 2^p per dimension')

        # invoke super init, passing number of dofs, dtype_u and dtype_f
        super().__init__((nvars, None, np.dtype('float64')))
        self._makeAttributeAndRegister(
            'nvars',
            'nu',
            'eps',
            'radius',
            'order',
            'useGPU',
            localVars=locals(),
            readOnly=True,
        )
        self._makeAttributeAndRegister(
            'newton_maxiter',
            'newton_tol',
            'lin_tol',
            'lin_maxiter',
            'inexact_linear_ratio',
            localVars=locals(),
            readOnly=False,
        )

        # compute dx and get discretization matrix A
        self.dx = 1.0 / self.nvars[0]
        self.A, _ = problem_helper.get_finite_difference_matrix(
            derivative=2,
            order=self.order,
            stencil_type='center',
            dx=self.dx,
            size=self.nvars[0],
            dim=2,
            bc='periodic',
            cupy=self.useGPU,
        )
        self.xvalues = self.xp.arange(self.nvars[0]) * self.dx - 0.5

        self.newton_ncalls = 0
        self.lin_ncalls = 0

        self.work_counters['newton'] = WorkCounter()
        self.work_counters['rhs'] = WorkCounter()
        self.work_counters['linear'] = WorkCounter()

    def reaction(self, u):
        r"""
        The reaction term, :math:`\frac{1}{2\varepsilon^2}(2u - 1)\left(1 - (2u - 1)^\nu\right)`.

        The wells sit at :math:`u = 0` and :math:`u = 1`, so the double well is symmetric about
        :math:`2u - 1`; writing the term in that variable is what lets :math:`\nu` keep the meaning
        it has always had here. For the default :math:`\nu = 2` this is
        :math:`-\frac{2}{\varepsilon^2} u (1 - u)(1 - 2u)`.
        """
        v = 2.0 * u - 1.0
        return 0.5 / self.eps**2 * v * (1.0 - v**self.nu)

    def reaction_prime(self, u):
        """Derivative of :meth:`reaction`, ready to go on the diagonal of a Jacobian."""
        v = 2.0 * u - 1.0
        return 1.0 / self.eps**2 * (1.0 - (self.nu + 1.0) * v**self.nu)

    def reaction_cubic(self, u):
        r"""The stiff part of :meth:`reaction`, :math:`-\frac{1}{2\varepsilon^2}(2u - 1)^{\nu + 1}`."""
        return -0.5 / self.eps**2 * (2.0 * u - 1.0) ** (self.nu + 1)

    def reaction_cubic_prime(self, u):
        """Derivative of :meth:`reaction_cubic`."""
        return -(self.nu + 1.0) / self.eps**2 * (2.0 * u - 1.0) ** self.nu

    def reaction_linear(self, u):
        r"""The rest of :meth:`reaction`, :math:`\frac{1}{2\varepsilon^2}(2u - 1)`, so the two sum back to it."""
        return 0.5 / self.eps**2 * (2.0 * u - 1.0)

    # noinspection PyTypeChecker
    def solve_system(self, rhs, factor, u0, t):
        """
        Simple Newton solver.

        Parameters
        ----------
        rhs : dtype_f
            Right-hand side for the nonlinear system
        factor : float
            Abbrev. for the node-to-node stepsize (or any other factor required).
        u0 : dtype_u
            Initial guess for the iterative solver.
        t : float
            Current time (required here for the BC).

        Returns
        -------
        me : dtype_u
            The solution as mesh.
        """

        # plain arrays, not the datatype: its `__abs__` is a norm, which CuPy's `linalg.norm` trips over
        u = u0.view(self.xp.ndarray).flatten()
        b = rhs.view(self.xp.ndarray).flatten()
        z = self.xp.zeros_like(u)

        Id = self.xsp.eye(self.nvars[0] * self.nvars[1])

        # start newton iteration
        n = 0
        res = 99
        while n < self.newton_maxiter:
            # form the function g with g(u) = 0
            g = u - factor * (self.A.dot(u) + self.reaction(u)) - b

            # if g is close to 0, then we are done
            res = self.xp.linalg.norm(g, self.xp.inf)

            # do inexactness in the linear solver
            if self.inexact_linear_ratio:
                self.lin_tol = res * self.inexact_linear_ratio

            if res < self.newton_tol:
                break

            # assemble dg
            dg = Id - factor * (self.A + self.xsp.diags(self.reaction_prime(u), offsets=0))

            # newton update: u1 = u0 - g/dg
            # u -= spsolve(dg, g)
            u -= self.linalg.cg(
                dg, g, x0=z, rtol=self.lin_tol, maxiter=self.lin_maxiter, atol=0, callback=self.work_counters['linear']
            )[0]
            # increase iteration count
            n += 1
            # print(n, res)

            self.work_counters['newton']()

        # if n == self.newton_maxiter:
        #     raise ProblemError('Newton did not converge after %i iterations, error is %s' % (n, res))

        me = self.dtype_u(self.init)
        me[:] = u.reshape(self.nvars)

        self.newton_ncalls += 1

        return me

    def eval_f(self, u, t):
        """
        Routine to evaluate the right-hand side of the problem.

        Parameters
        ----------
        u : dtype_u
            Current values of the numerical solution.
        t : float
            Current time of the numerical solution is computed (not used here).

        Returns
        -------
        f : dtype_f
            The right-hand side of the problem.
        """
        f = self.dtype_f(self.init)
        v = u.flatten()
        f[:] = (self.A.dot(v) + self.reaction(v)).reshape(self.nvars)

        self.work_counters['rhs']()
        return f

    def u_exact(self, t, u_init=None, t_init=None):
        r"""
        Routine to compute the exact solution at time :math:`t`.

        Parameters
        ----------
        t : float
            Time of the exact solution.

        Returns
        -------
        me : dtype_u
            The exact solution.
        """
        me = self.dtype_u(self.init, val=0.0)
        if t > 0:

            def eval_rhs(t, u):
                return self.eval_f(u.reshape(self.init[0]), t).flatten()

            me[:] = self.generate_scipy_reference_solution(eval_rhs, t, u_init, t_init)

        else:
            X, Y = self.xp.meshgrid(self.xvalues, self.xvalues)
            r2 = X**2 + Y**2
            me[:] = 0.5 * (1.0 + self.xp.tanh((self.radius - self.xp.sqrt(r2)) / (np.sqrt(2) * self.eps)))

        return me


# noinspection PyUnusedLocal
class allencahn_semiimplicit(allencahn_fullyimplicit):
    r"""
    This class implements the two-dimensional Allen-Cahn equation with periodic boundary conditions, with the two
    phases at :math:`u = 0` and :math:`u = 1`

    .. math::
        \frac{\partial u}{\partial t} = \Delta u
            + \frac{1}{2\varepsilon^2} (2u - 1)\left(1 - (2u - 1)^\nu\right)

    for a constant :math:`\nu`, which at the default :math:`\nu = 2` is the usual
    :math:`\Delta u - \frac{2}{\varepsilon^2} u (1 - u)(1 - 2u)`.

    Initial condition are circles of the form

    .. math::
        u({\bf x}, 0) = \frac{1}{2}\left(1 + \tanh\left(\frac{r - \sqrt{x_i^2 + y_j^2}}
        {\sqrt{2}\varepsilon}\right)\right)

    for :math:`i, j=0,..,N-1`, where :math:`N` is the number of spatial grid points. For time-stepping, the problem is
    treated in a *semi-implicit* way, i.e., the linear system containing the Laplacian is solved by the conjugate gradients
    method, and the system containing the rest of the right-hand side is only evaluated at each time.
    """

    dtype_f = imex_mesh

    def eval_f(self, u, t):
        """
        Routine to evaluate the right-hand side of the problem.

        Parameters
        ----------
        u : dtype_u
            Current values of the numerical solution.
        t : float
            Current time of the numerical solution is computed (not used here).

        Returns
        -------
        f : dtype_f
            The right-hand side of the problem.
        """
        f = self.dtype_f(self.init)
        v = u.flatten()
        f.impl[:] = self.A.dot(v).reshape(self.nvars)
        f.expl[:] = self.reaction(v).reshape(self.nvars)

        self.work_counters['rhs']()
        return f

    def solve_system(self, rhs, factor, u0, t):
        r"""
        Simple linear solver for :math:`(I-factor\cdot A)\vec{u}=\vec{rhs}`.

        Parameters
        ----------
        rhs : dtype_f
            Right-hand side for the linear system.
        factor : float
            Abbrev. for the local stepsize (or any other factor required).
        u0 : dtype_u
            Initial guess for the iterative solver.
        t : float
            Current time (e.g. for time-dependent BCs).

        Returns
        -------
        me : dtype_u
            The solution as mesh.
        """

        me = self.dtype_u(self.init)

        Id = self.xsp.eye(self.nvars[0] * self.nvars[1])

        me[:] = self.linalg.cg(
            Id - factor * self.A,
            rhs.flatten(),
            x0=u0.flatten(),
            rtol=self.lin_tol,
            maxiter=self.lin_maxiter,
            atol=0,
            callback=self.work_counters['linear'],
        )[0].reshape(self.nvars)

        self.lin_ncalls += 1

        return me

    def u_exact(self, t, u_init=None, t_init=None):
        """
        Routine to compute the exact solution at time t.

        Parameters
        ----------
        t : float
            Time of the exact solution.

        Returns
        -------
        me : dtype_u
            The exact solution.
        """
        me = self.dtype_u(self.init, val=0.0)
        if t > 0:

            def eval_rhs(t, u):
                f = self.eval_f(u.reshape(self.init[0]), t)
                return (f.impl + f.expl).flatten()

            me[:] = self.generate_scipy_reference_solution(eval_rhs, t, u_init, t_init)
        else:
            me[:] = super().u_exact(t, u_init, t_init)
        return me


# noinspection PyUnusedLocal
class allencahn_semiimplicit_v2(allencahn_fullyimplicit):
    r"""
    This class implements the two-dimensional Allen-Cahn (AC) equation with periodic boundary conditions, with the two
    phases at :math:`u = 0` and :math:`u = 1`

    .. math::
        \frac{\partial u}{\partial t} = \Delta u
            + \frac{1}{2\varepsilon^2} (2u - 1)\left(1 - (2u - 1)^\nu\right)

    for a constant :math:`\nu`, which at the default :math:`\nu = 2` is the usual
    :math:`\Delta u - \frac{2}{\varepsilon^2} u (1 - u)(1 - 2u)`.

    Initial condition are circles of the form

    .. math::
        u({\bf x}, 0) = \frac{1}{2}\left(1 + \tanh\left(\frac{r - \sqrt{x_i^2 + y_j^2}}
        {\sqrt{2}\varepsilon}\right)\right)

    for :math:`i, j=0,..,N-1`, where :math:`N` is the number of spatial grid points. For time-stepping, a special AC-splitting
    is used to get a *semi-implicit* treatment of the problem: The term :math:`\Delta u - \frac{1}{2\varepsilon^2}(2u - 1)^3`
    is handled implicitly and the nonlinear system including this part will be solved by Newton. :math:`\frac{1}{2\varepsilon^2}(2u - 1)`
    is only evaluated at each time.
    """

    dtype_f = imex_mesh

    def eval_f(self, u, t):
        """
        Routine to evaluate the right-hand side of the problem.

        Parameters
        ----------
        u : dtype_u
            Current values of the numerical solution.
        t : float
            Current time of the numerical solution is computed.

        Returns
        -------
        f : dtype_f
            The right-hand side of the problem.
        """
        f = self.dtype_f(self.init)
        v = u.flatten()
        f.impl[:] = (self.A.dot(v) + self.reaction_cubic(v)).reshape(self.nvars)
        f.expl[:] = self.reaction_linear(v).reshape(self.nvars)

        self.work_counters['rhs']()
        return f

    def solve_system(self, rhs, factor, u0, t):
        """
        Simple Newton solver.

        Parameters
        ----------
        rhs : dtype_f
            Right-hand side for the nonlinear system.
        factor : float
            Abbrev. for the node-to-node stepsize (or any other factor required).
        u0 : dtype_u
            Initial guess for the iterative solver.
        t : float
            Current time (required here for the BC).

        Returns
        -------
        me : dtype_u
            The solution as mesh.
        """

        # plain arrays, not the datatype: its `__abs__` is a norm, which CuPy's `linalg.norm` trips over
        u = u0.view(self.xp.ndarray).flatten()
        b = rhs.view(self.xp.ndarray).flatten()
        z = self.xp.zeros_like(u)

        Id = self.xsp.eye(self.nvars[0] * self.nvars[1])

        # start newton iteration
        n = 0
        res = 99
        while n < self.newton_maxiter:
            # form the function g with g(u) = 0
            g = u - factor * (self.A.dot(u) + self.reaction_cubic(u)) - b

            # if g is close to 0, then we are done
            res = self.xp.linalg.norm(g, self.xp.inf)

            if res < self.newton_tol:
                break

            # assemble dg
            dg = Id - factor * (self.A + self.xsp.diags(self.reaction_cubic_prime(u), offsets=0))

            # newton update: u1 = u0 - g/dg
            # u -= spsolve(dg, g)
            u -= self.linalg.cg(dg, g, x0=z, rtol=self.lin_tol, atol=0)[0]
            # increase iteration count
            n += 1
            # print(n, res)

            self.work_counters['newton']()

        # if n == self.newton_maxiter:
        #     raise ProblemError('Newton did not converge after %i iterations, error is %s' % (n, res))

        me = self.dtype_u(self.init)
        me[:] = u.reshape(self.nvars)

        self.newton_ncalls += 1

        return me


# noinspection PyUnusedLocal
class allencahn_multiimplicit(allencahn_fullyimplicit):
    r"""
    Example implementing the two-dimensional Allen-Cahn equation with periodic boundary conditions, with the two
    phases at :math:`u = 0` and :math:`u = 1`

    .. math::
        \frac{\partial u}{\partial t} = \Delta u
            + \frac{1}{2\varepsilon^2} (2u - 1)\left(1 - (2u - 1)^\nu\right)

    for a constant :math:`\nu`, which at the default :math:`\nu = 2` is the usual
    :math:`\Delta u - \frac{2}{\varepsilon^2} u (1 - u)(1 - 2u)`.

    Initial condition are circles of the form

    .. math::
        u({\bf x}, 0) = \frac{1}{2}\left(1 + \tanh\left(\frac{r - \sqrt{x_i^2 + y_j^2}}
        {\sqrt{2}\varepsilon}\right)\right)

    for :math:`i, j=0,..,N-1`, where :math:`N` is the number of spatial grid points. For time-stepping, the problem is
    treated in *multi-implicit* fashion, i.e., the linear system containing the Laplacian is solved by the conjugate gradients
    method, and the system containing the rest of the right-hand side will be solved by Newton's method.
    """

    dtype_f = comp2_mesh

    def eval_f(self, u, t):
        """
        Routine to evaluate the right-hand side of the problem.

        Parameters
        ----------
        u : dtype_u
            Current values of the numerical solution.
        t : float
            Current time of the numerical solution is computed.

        Returns
        -------
        f : dtype_f
            The right-hand side of the problem.
        """
        f = self.dtype_f(self.init)
        v = u.flatten()
        f.comp1[:] = self.A.dot(v).reshape(self.nvars)
        f.comp2[:] = self.reaction(v).reshape(self.nvars)

        self.work_counters['rhs']()
        return f

    def solve_system_1(self, rhs, factor, u0, t):
        r"""
        Simple linear solver for :math:`(I-factor\cdot A)\vec{u}=\vec{rhs}`.

        Parameters
        ----------
        rhs : dtype_f
            Right-hand side for the linear system.
        factor : float
            Abbrev. for the local stepsize (or any other factor required).
        u0 : dtype_u
            Initial guess for the iterative solver.
        t : float
            Current time (e.g. for time-dependent BCs).

        Returns
        -------
        me : dtype_u
            The solution as mesh.
        """

        me = self.dtype_u(self.init)

        Id = self.xsp.eye(self.nvars[0] * self.nvars[1])

        me[:] = self.linalg.cg(
            Id - factor * self.A,
            rhs.flatten(),
            x0=u0.flatten(),
            rtol=self.lin_tol,
            maxiter=self.lin_maxiter,
            atol=0,
            callback=self.work_counters['linear'],
        )[0].reshape(self.nvars)

        self.lin_ncalls += 1

        return me

    def solve_system_2(self, rhs, factor, u0, t):
        """
        Simple Newton solver.

        Parameters
        ----------
        rhs : dtype_f
            Right-hand side for the nonlinear system.
        factor : float
            Abbrev. for the node-to-node stepsize (or any other factor required).
        u0 : dtype_u
            Initial guess for the iterative solver.
        t : float
            Current time (required here for the BC).

        Returns
        -------
        me : dtype_u
            The solution as mesh.
        """

        # plain arrays, not the datatype: its `__abs__` is a norm, which CuPy's `linalg.norm` trips over
        u = u0.view(self.xp.ndarray).flatten()
        b = rhs.view(self.xp.ndarray).flatten()
        z = self.xp.zeros_like(u)

        Id = self.xsp.eye(self.nvars[0] * self.nvars[1])

        # start newton iteration
        n = 0
        res = 99
        while n < self.newton_maxiter:
            # form the function g with g(u) = 0
            g = u - factor * self.reaction(u) - b

            # if g is close to 0, then we are done
            res = self.xp.linalg.norm(g, self.xp.inf)

            if res < self.newton_tol:
                break

            # assemble dg
            dg = Id - factor * self.xsp.diags(self.reaction_prime(u), offsets=0)

            # newton update: u1 = u0 - g/dg
            # u -= spsolve(dg, g)
            u -= self.linalg.cg(dg, g, x0=z, rtol=self.lin_tol, atol=0)[0]
            # increase iteration count
            n += 1
            # print(n, res)

            self.work_counters['newton']()

        # if n == self.newton_maxiter:
        #     raise ProblemError('Newton did not converge after %i iterations, error is %s' % (n, res))

        me = self.dtype_u(self.init)
        me[:] = u.reshape(self.nvars)

        self.newton_ncalls += 1

        return me


# noinspection PyUnusedLocal
class allencahn_multiimplicit_v2(allencahn_fullyimplicit):
    r"""
    This class implements the two-dimensional Allen-Cahn (AC) equation with periodic boundary conditions, with the two
    phases at :math:`u = 0` and :math:`u = 1`

    .. math::
        \frac{\partial u}{\partial t} = \Delta u
            + \frac{1}{2\varepsilon^2} (2u - 1)\left(1 - (2u - 1)^\nu\right)

    for a constant :math:`\nu`, which at the default :math:`\nu = 2` is the usual
    :math:`\Delta u - \frac{2}{\varepsilon^2} u (1 - u)(1 - 2u)`.

    The initial condition has the form of circles

    .. math::
        u({\bf x}, 0) = \frac{1}{2}\left(1 + \tanh\left(\frac{r - \sqrt{x_i^2 + y_j^2}}
        {\sqrt{2}\varepsilon}\right)\right)

    for :math:`i, j=0,..,N-1`, where :math:`N` is the number of spatial grid points. For time-stepping, a special AC-splitting
    is used here to get another kind of *semi-implicit* treatment of the problem: The term :math:`\Delta u - \frac{1}{2\varepsilon^2}(2u - 1)^3`
    is handled implicitly and the nonlinear system including this part will be solved by Newton. :math:`\frac{1}{2\varepsilon^2}(2u - 1)`
    is solved by a linear solver provided by a ``SciPy`` routine.
    """

    dtype_f = comp2_mesh

    def eval_f(self, u, t):
        """
        Routine to evaluate the right-hand side of the problem.

        Parameters
        ----------
        u : dtype_u
            Current values of the numerical solution.
        t : float
            Current time of the numerical solution is computed.

        Returns
        -------
        f : dtype_f
            The right-hand side of the problem.
        """
        f = self.dtype_f(self.init)
        v = u.flatten()
        f.comp1[:] = (self.A.dot(v) + self.reaction_cubic(v)).reshape(self.nvars)
        f.comp2[:] = self.reaction_linear(v).reshape(self.nvars)

        self.work_counters['rhs']()
        return f

    def solve_system_1(self, rhs, factor, u0, t):
        """
        Simple Newton solver.

        Parameters
        ----------
        rhs : dtype_f
            Right-hand side for the nonlinear system.
        factor : float
            Abbrev. for the node-to-node stepsize (or any other factor required).
        u0 : dtype_u
            Initial guess for the iterative solver.
        t : float
            Current time (required here for the BC).

        Returns
        ------
        me : dtype_u
            The solution as mesh.
        """

        # plain arrays, not the datatype: its `__abs__` is a norm, which CuPy's `linalg.norm` trips over
        u = u0.view(self.xp.ndarray).flatten()
        b = rhs.view(self.xp.ndarray).flatten()
        z = self.xp.zeros_like(u)

        Id = self.xsp.eye(self.nvars[0] * self.nvars[1])

        # start newton iteration
        n = 0
        res = 99
        while n < self.newton_maxiter:
            # form the function g with g(u) = 0
            g = u - factor * (self.A.dot(u) + self.reaction_cubic(u)) - b

            # if g is close to 0, then we are done
            res = self.xp.linalg.norm(g, self.xp.inf)

            if res < self.newton_tol:
                break

            # assemble dg
            dg = Id - factor * (self.A + self.xsp.diags(self.reaction_cubic_prime(u), offsets=0))

            # newton update: u1 = u0 - g/dg
            # u -= spsolve(dg, g)
            u -= self.linalg.cg(
                dg,
                g,
                x0=z,
                rtol=self.lin_tol,
                atol=0,
            )[0]
            # increase iteration count
            n += 1
            # print(n, res)

            self.work_counters['newton']()

        # if n == self.newton_maxiter:
        #     raise ProblemError('Newton did not converge after %i iterations, error is %s' % (n, res))

        me = self.dtype_u(self.init)
        me[:] = u.reshape(self.nvars)

        self.newton_ncalls += 1

        return me

    def solve_system_2(self, rhs, factor, u0, t):
        r"""
        Simple linear solver for :math:`(I-factor\cdot A)\vec{u}=\vec{rhs}`.

        Parameters
        ----------
        rhs : dtype_f
            Right-hand side for the linear system.
        factor : float
            Abbrev. for the local stepsize (or any other factor required).
        u0 : dtype_u
            Initial guess for the iterative solver.
        t : float
            Current time (e.g. for time-dependent BCs).

        Returns
        -------
        me : dtype_u
            The solution as mesh.
        """

        me = self.dtype_u(self.init)

        me[:] = ((rhs - 0.5 * factor / self.eps**2) / (1.0 - factor / self.eps**2)).reshape(self.nvars)
        return me
