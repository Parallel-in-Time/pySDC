"""
Derived problem classes for the Allen-Cahn 1D FD order-reduction playground.

This module defines two problem classes that extend
:class:`~pySDC.implementations.problem_classes.AllenCahn_1D_FD.allencahn_front_fullyimplicit`
to demonstrate SDC order reduction with time-dependent Dirichlet boundary
conditions and its remedy via **boundary lifting**.

Background
----------
The Allen-Cahn equation on :math:`[-0.5, 0.5]` with a travelling-wave exact
solution

.. math::
    u^*(x, t) = \\frac{1}{2}\\left(1 + \\tanh\\frac{x - vt}{\\sqrt{2}\\,\\varepsilon}\\right),
    \\qquad v = 3\\sqrt{2}\\,\\varepsilon\\,\\delta w,

carries **time-dependent Dirichlet boundary conditions**

.. math::
    g_{L/R}(t) = u^*(x_{L/R}, t).

When these BCs are imposed inside ``solve_system`` of the Newton iteration,
the fixed point of the SDC sweep no longer matches the exact collocation
solution, causing **order reduction**: the observed convergence order is lower
than the theoretical SDC order :math:`2M - 1`.

**Boundary lifting** removes the source of reduction by introducing the new
variable

.. math::
    w = u - E,

where :math:`E(x, t)` is a linear-in-:math:`x` lift function that already
satisfies the time-dependent BCs:

.. math::
    E(x, t) = g_L(t)\\,\\frac{x_R - x}{x_R - x_L} + g_R(t)\\,\\frac{x - x_L}{x_R - x_L}.

Because :math:`E_{xx} = 0`, the equation for :math:`w` is

.. math::
    w_t = w_{xx} - \\frac{2}{\\varepsilon^2} u(1-u)(1-2u) - 6\\delta w\\,u(1-u) - E_t,

with **homogeneous** Dirichlet BCs :math:`w(x_{L/R}, t) = 0`.  The standard
Newton-based ``solve_system`` then sees zero BCs throughout, and the full SDC
convergence order is restored.

Note on the time-step constraint
---------------------------------
The Allen-Cahn nonlinear term introduces a stiffness :math:`O(1/\\varepsilon^2)`.
For semiimplicit (IMEX) discretisations this imposes the stability constraint
:math:`\\Delta t \\lesssim \\varepsilon^2`.  With a **fully implicit** Newton
solver the constraint disappears, but it still defines the natural time scale
of the problem.  The playground uses :math:`\\varepsilon = 0.5` so that
:math:`\\varepsilon^2 = 0.25` is a convenient upper bound for :math:`\\Delta t`,
keeping the solver in the regime where temporal errors dominate over the
spatial (FD) discretisation error.
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

from pySDC.core.errors import ProblemError
from pySDC.implementations.problem_classes.AllenCahn_1D_FD import allencahn_front_fullyimplicit


class allencahn_front_fullyimplicit_naive(allencahn_front_fullyimplicit):
    r"""
    Allen-Cahn 1D fully-implicit problem with **zero BCs** inside ``solve_system``.

    This class is identical to :class:`allencahn_front_fullyimplicit` except
    that ``solve_system`` uses homogeneous (zero) Dirichlet boundary conditions
    in the Newton iteration regardless of the actual time-dependent BCs.

    The mismatch between the BCs used in ``eval_f`` (correct, time-dependent)
    and those used in ``solve_system`` (zero) means the fixed point of the SDC
    sweep does **not** match the collocation solution.  The result is
    **order reduction**: the numerical convergence order is much lower than
    the theoretical SDC order :math:`2M - 1`.

    This class is intended as the "naive" reference that exhibits order
    reduction, to be compared with :class:`allencahn_front_fullyimplicit_lift`
    which restores the full order via boundary lifting.

    Parameters
    ----------
    Same as :class:`allencahn_front_fullyimplicit`.
    """

    def solve_system(self, rhs, factor, u0, t):
        r"""
        Newton solver with **zero Dirichlet BCs** — the source of order reduction.

        The Newton residual uses ``self.uext[0] = self.uext[-1] = 0`` instead
        of the exact boundary values :math:`g_{L/R}(t)`.  This introduces a
        per-step boundary error of order :math:`O(\Delta t)`, which prevents
        the SDC iteration from converging to the true collocation solution and
        reduces the global convergence order.

        Parameters
        ----------
        rhs : dtype_u
            Right-hand side for the nonlinear system.
        factor : float
            Implicit factor (node-to-node step size or similar).
        u0 : dtype_u
            Initial guess for Newton's method.
        t : float
            Current time (used only to match the interface; BCs are set to zero).

        Returns
        -------
        me : dtype_u
            Approximate solution with zero-BC Newton solve.
        """
        u = self.dtype_u(u0)
        eps2 = self.eps**2
        dw = self.dw

        Id = sp.eye(self.nvars)
        A = self.A[1:-1, 1:-1]

        # NAIVE: use zero boundary conditions (wrong — the true BCs are time-dependent)
        self.uext[0] = 0.0
        self.uext[-1] = 0.0

        n = 0
        res = 99
        while n < self.newton_maxiter:
            self.uext[1:-1] = u[:]
            g = u - rhs - factor * (
                self.A.dot(self.uext)[1:-1]
                - 2.0 / eps2 * u * (1.0 - u) * (1.0 - 2.0 * u)
                - 6.0 * dw * u * (1.0 - u)
            )
            res = np.linalg.norm(g, np.inf)
            if res < self.newton_tol:
                break
            dg = Id - factor * (
                A
                - 2.0 / eps2 * sp.diags((1.0 - u) * (1.0 - 2.0 * u) - u * ((1.0 - 2.0 * u) + 2.0 * (1.0 - u)))
                - 6.0 * dw * sp.diags((1.0 - u) - u)
            )
            u -= spsolve(dg, g)
            n += 1
            self.work_counters['newton']()

        if np.isnan(res) and self.stop_at_nan:
            raise ProblemError(f'Newton got nan after {n} iterations, aborting...')
        elif np.isnan(res):
            self.logger.warning(f'Newton got nan after {n} iterations...')

        if n == self.newton_maxiter:
            msg = f'Newton did not converge after {n} iterations, error is {res}'
            if self.stop_at_maxiter:
                raise ProblemError(msg)
            else:
                self.logger.warning(msg)

        me = self.dtype_u(self.init)
        me[:] = u[:]
        return me


class allencahn_front_fullyimplicit_lift(allencahn_front_fullyimplicit):
    r"""
    Allen-Cahn 1D fully-implicit problem with **boundary lifting**.

    Instead of solving for :math:`u` directly, this class solves for the
    lifted variable

    .. math::
        w = u - E,

    where the linear lift

    .. math::
        E(x, t) = g_L(t)\,\frac{x_R - x}{x_R - x_L}
                 + g_R(t)\,\frac{x - x_L}{x_R - x_L}

    interpolates the time-dependent boundary data
    :math:`g_L(t) = u^*(x_L, t)` and :math:`g_R(t) = u^*(x_R, t)`.

    Because :math:`w = 0` at the boundaries for all :math:`t`, the Newton
    solve inside ``solve_system`` uses **homogeneous** (zero) Dirichlet BCs,
    which is now correct.  This eliminates the source of order reduction and
    the full SDC convergence order :math:`2M - 1` is restored.

    The equation for :math:`w` reads

    .. math::
        w_t = w_{xx}
              - \frac{2}{\varepsilon^2} u(1-u)(1-2u)
              - 6\delta w\, u(1-u)
              - E_t(x, t),

    where :math:`u = w + E` and :math:`E_t` is the time derivative of the
    lift.  The Jacobian used in Newton's method is identical in structure to
    the original problem, but evaluated at :math:`u = w + E` with the
    interior-only stiffness matrix.

    Parameters
    ----------
    Same as :class:`allencahn_front_fullyimplicit`.
    """

    def _compute_lift(self, t):
        r"""
        Compute the lift :math:`E(x, t)` and its time derivative :math:`E_t(x, t)`.

        The lift is the unique linear-in-:math:`x` function satisfying
        :math:`E(x_L, t) = g_L(t)` and :math:`E(x_R, t) = g_R(t)`, where
        :math:`g_{L/R}(t) = u^*(x_{L/R}, t)` are the exact boundary values.

        Parameters
        ----------
        t : float
            Current time.

        Returns
        -------
        E : numpy.ndarray, shape (nvars,)
            Lift values at interior grid points.
        E_t : numpy.ndarray, shape (nvars,)
            Time derivative of the lift at interior grid points.
        """
        v = 3.0 * np.sqrt(2) * self.eps * self.dw
        sq2eps = np.sqrt(2) * self.eps

        g_L = 0.5 * (1.0 + np.tanh((self.interval[0] - v * t) / sq2eps))
        g_R = 0.5 * (1.0 + np.tanh((self.interval[1] - v * t) / sq2eps))

        L = self.interval[1] - self.interval[0]
        xi = (self.xvalues - self.interval[0]) / L  # ∈ [0, 1]

        E = g_L * (1.0 - xi) + g_R * xi

        # Time derivatives: dg/dt = -v/(2*sq2eps) * sech²(...)
        sech2_L = 1.0 - np.tanh((self.interval[0] - v * t) / sq2eps) ** 2
        sech2_R = 1.0 - np.tanh((self.interval[1] - v * t) / sq2eps) ** 2
        g_L_t = -v / (2.0 * sq2eps) * sech2_L
        g_R_t = -v / (2.0 * sq2eps) * sech2_R

        E_t = g_L_t * (1.0 - xi) + g_R_t * xi

        return E, E_t

    def eval_f(self, w, t):
        r"""
        Evaluate the right-hand side of the lifted equation.

        Given the lifted variable :math:`w`, recovers :math:`u = w + E` and
        evaluates

        .. math::
            f(w, t) = w_{xx}
                      - \frac{2}{\varepsilon^2} u(1-u)(1-2u)
                      - 6\delta w\, u(1-u)
                      - E_t.

        The Laplacian :math:`w_{xx}` is computed from the full stiffness matrix
        :math:`A` with zero ghost-point values (consistent with :math:`w = 0`
        at the boundaries).

        Parameters
        ----------
        w : dtype_u
            Current lifted variable.
        t : float
            Current time.

        Returns
        -------
        f : dtype_f
            Right-hand side of the lifted equation.
        """
        E, E_t = self._compute_lift(t)
        u = w[:] + E

        # Laplacian of w with zero BCs (correct since w=0 at boundaries)
        self.uext[0] = 0.0
        self.uext[-1] = 0.0
        self.uext[1:-1] = w[:]

        f = self.dtype_f(self.init)
        f[:] = (
            self.A.dot(self.uext)[1:-1]  # w_xx (zero BCs)
            - 2.0 / self.eps**2 * u * (1.0 - u) * (1.0 - 2.0 * u)
            - 6.0 * self.dw * u * (1.0 - u)
            - E_t
        )
        self.work_counters['rhs']()
        return f

    def solve_system(self, rhs, factor, w0, t):
        r"""
        Newton solver for the lifted equation with **zero Dirichlet BCs**.

        Solves :math:`w - \text{factor}\cdot f(w, t) = \text{rhs}` where the
        Newton residual and Jacobian are evaluated at :math:`u = w + E(t)`.
        Since :math:`w = 0` at the boundaries, zero Dirichlet BCs are
        **correct** here, eliminating the source of order reduction.

        Parameters
        ----------
        rhs : dtype_u
            Right-hand side for the Newton system.
        factor : float
            Implicit factor.
        w0 : dtype_u
            Initial guess.
        t : float
            Current time.

        Returns
        -------
        me : dtype_u
            Solution of the lifted Newton system.
        """
        w = self.dtype_u(w0)
        E, E_t = self._compute_lift(t)

        eps2 = self.eps**2
        dw_param = self.dw
        Id = sp.eye(self.nvars)
        A_inner = self.A[1:-1, 1:-1]  # interior-only stiffness (zero BCs for w)

        n = 0
        res = 99
        while n < self.newton_maxiter:
            u = w[:] + E
            # Residual: g(w) = w - rhs - factor * f(w, t)
            fw = (
                A_inner.dot(w)
                - 2.0 / eps2 * u * (1.0 - u) * (1.0 - 2.0 * u)
                - 6.0 * dw_param * u * (1.0 - u)
                - E_t
            )
            g = w - rhs - factor * fw
            res = np.linalg.norm(g, np.inf)
            if res < self.newton_tol:
                break
            # Jacobian: dg/dw = I - factor * (A_inner + d(nonlinear)/du * du/dw)
            # du/dw = 1 since u = w + E(t) with E independent of w
            dg = Id - factor * (
                A_inner
                - 2.0 / eps2 * sp.diags(
                    (1.0 - u) * (1.0 - 2.0 * u) - u * ((1.0 - 2.0 * u) + 2.0 * (1.0 - u))
                )
                - 6.0 * dw_param * sp.diags((1.0 - u) - u)
            )
            w -= spsolve(dg, g)
            n += 1
            self.work_counters['newton']()

        if np.isnan(res) and self.stop_at_nan:
            raise ProblemError(f'Newton got nan after {n} iterations, aborting...')
        elif np.isnan(res):
            self.logger.warning(f'Newton got nan after {n} iterations...')

        if n == self.newton_maxiter:
            msg = f'Newton did not converge after {n} iterations, error is {res}'
            if self.stop_at_maxiter:
                raise ProblemError(msg)
            else:
                self.logger.warning(msg)

        me = self.dtype_u(self.init)
        me[:] = w[:]
        return me

    def u_exact(self, t):
        r"""
        Exact solution of the lifted variable :math:`w = u^* - E` at time :math:`t`.

        Parameters
        ----------
        t : float
            Time at which the exact lifted solution is computed.

        Returns
        -------
        me : dtype_u
            Exact lifted solution :math:`w^*(x, t) = u^*(x, t) - E(x, t)`.
        """
        E, _ = self._compute_lift(t)
        u_star = allencahn_front_fullyimplicit.u_exact(self, t)
        me = self.dtype_u(self.init, val=0.0)
        me[:] = u_star[:] - E
        return me
