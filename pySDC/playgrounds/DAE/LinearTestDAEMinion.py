import numpy as np

from pySDC.core.problem import WorkCounter
from pySDC.projects.DAE.misc.ProblemDAE import ptype_dae
from pySDC.core.errors import ProblemError


class LinearTestDAEMinion(ptype_dae):
    r"""
    This class implements a linear stiff DAE problem from [1]_ that is given by

    .. math::
        \frac{d}{dt} u_1 = u_1 - u_3 + u_4,

    .. math::
        \frac{d}{dt} u_2 = -10^4 u_2 + (1 + 10^4) e^t, 

    .. math::
        \frac{d}{dt} u_3 = u_1,

    .. math::
        0 = u_1 + u_2 + u_4 - e^t

    for :math:`0 < \varepsilon \ll 1`. The linear system at each node is solved by Newton's method. Note
    that the system can also be solved directly by a linear solver.

    References
    ----------
    .. [1] S. Bu, J. Huang, M. L. Minion. Semi-implicit Krylov deferred correction methods for differential
           algebraic equations. Math. Comput. 81, No. 280, 2127-2157 (2012).
    """

    def __init__(self, newton_tol=1e-12):
        """Initialization routine"""
        super().__init__(nvars=(3, 1), newton_tol=newton_tol)
        self._makeAttributeAndRegister('newton_tol', localVars=locals())
        self.work_counters['rhs'] = WorkCounter()
        self.work_counters['newton'] = WorkCounter()

    def eval_f(self, u, du, t):
        r"""
        Routine to evaluate the implicit representation of the problem, i.e., :math:`F(u, u', t)`.

        Parameters
        ----------
        u : dtype_u
            Current values of the numerical solution at time t.
        du : dtype_u
            Current values of the derivative of the numerical solution at time t.
        t : float
            Current time of the numerical solution.

        Returns
        -------
        f : dtype_f
            The right-hand side of f (contains four components).
        """

        u1, u2, u3 = u.diff[0], u.diff[1], u.diff[2]
        du1, du2, du3 = du.diff[0], du.diff[1], du.diff[2]
        u4 = u.alg[0]

        f = self.dtype_f(self.init)
        f.diff[0] = u1 - u3 + u4 - du1
        f.diff[1] = -1e4 * u2 + (1 + 1e4) * np.exp(t) - du2
        f.diff[2] = u1 - du3
        f.alg[0] = u1 + u2 + u4 - np.exp(t)

    def u_exact(self, t):
        r"""
        Routine for the exact solution at time :math:`t`.

        Parameters
        ----------
        t : float
            Time of the exact solution.

        Returns
        -------
        me : dtype_u
            Exact solution.
        """

        me = self.dtype_u(self.init)
        me.diff[0] = np.cos(t)
        me.diff[1] = np.exp(t)
        me.diff[2] = np.sin(t)
        me.alg[0] = -np.cos(t)
        return me


class LinearTestDAEMinionConstrained(LinearTestDAEMinion):
    r"""
    In this class the example ``LinearTestDAEMinion`` is implemented in the way that the system to be solved on each node
    quadrature is only applied to the differential parts. In order to solve this problem ``genericImplicitDAEConstrained``
    needs to be used.

    Note
    ----
    The problem is of the general form

    .. math::
        \begin{pmatrix}
            u_1'\\
            u_2'\\
            u_3'\\
            0
        \end{pmatrix} = \begin{pmatrix}
            1 & 0 & -1 & 1
            0 & -10^4 & 0 & 0 \\
            1 & 0 & 0 & 0 \\
            1 & 1 & 0 & 1
        \end{pmatrix}\begin{pmatrix}
            u_1\\
            u_2\\
            u_3\\
            u_4
        \end{pmatrix} + \begin{pmatrix}
            0 \\
            (1 + 10^4) e^t \\
            0 \\
            -e^t
        \end{pmatrix}

    and it could be think about to also treat this problem in a semi-implicit way.
    """

    def __init__(self, nvars=(3, 1), newton_tol=1e-12, newton_maxiter=100, stop_at_maxiter=False, stop_at_nan=True):
        """Initialization routine"""
        super().__init__()
        self._makeAttributeAndRegister('newton_tol', 'newton_maxiter', 'stop_at_maxiter', 'stop_at_nan', localVars=locals())
        self.work_counters['newton'] = WorkCounter()
        self.work_counters['rhs'] = WorkCounter()

    def eval_f(self, u, t):
        r"""
        Routine to evaluate the right-hand side of the problem.

        Parameters
        ----------
        u : dtype_u
            Current values of the numerical solution at time t.
        t : float
            Current time of the numerical solution.

        Returns
        -------
        f : dtype_f
            The right-hand side of f (contains four components).
        """

        u1, u2, u3 = u.diff[0], u.diff[1], u.diff[2]
        u4 = u.alg[0]

        f = self.dtype_f(self.init)
        f.diff[0] = u1 - u3 + u4
        f.diff[1] = -1e4 * u2 + (1 + 1e4) * np.exp(t)
        f.diff[2] = u1
        f.alg[0] = u1 + u2 + u4 - np.exp(t)
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

        u = self.dtype_u(u0)

        # start newton iteration
        n = 0
        res = 99
        while n < self.newton_maxiter:
            u1, u2, u3 = u.diff[0], u.diff[1], u.diff[2]

            f = self.eval_f(u, t)

            # form the function g(u), such that the solution to the nonlinear problem is a root of g
            g = np.array(
                [
                    u1 - factor * f.diff[0] - rhs.diff[0],
                    u2 - factor * f.diff[1] - rhs.diff[1],
                    u3 - factor * f.diff[2] - rhs.diff[2],
                    f.alg[0],
                ]
            )

            # if g is close to 0, then we are done
            res = np.linalg.norm(g, np.inf)
            if res < self.newton_tol:
                break

            # assemble dg
            dg = np.array(
                [
                    [1 - factor, 0, factor, -factor],
                    [0, 1 + factor * 1e4, 0, 0],
                    [-factor, 0, 1, 0],
                    [1, 1, 0, 1],
                ]
            )

            # newton update: u1 = u0 - g/dg
            dx = np.linalg.solve(dg, g)

            u.diff[0] -= dx[0]
            u.diff[1] -= dx[1]
            u.diff[2] -= dx[2]
            u.alg[0] -= dx[3]

            n += 1
            self.work_counters['newton']()

        if np.isnan(res) and self.stop_at_nan:
            raise ProblemError('Newton got nan after %i iterations, aborting...' % n)
        elif np.isnan(res):
            self.logger.warning('Newton got nan after %i iterations...' % n)

        if n == self.newton_maxiter:
            msg = 'Newton did not converge after %i iterations, error is %s' % (n, res)
            if self.stop_at_maxiter:
                raise ProblemError(msg)
            else:
                self.logger.warning(msg)

        me = self.dtype_u(self.init)
        me[:] = u[:]

        return me


class LinearTestDAEMinionEmbedded(LinearTestDAEMinionConstrained):
    r"""
    The problem in this class inherits from the example(s) above and can be solved using the embedded SDC scheme ``genericImplicitEmbedded``.
    ``solve_system`` is constructed so that the implicit system to be solved on each node uses neither the initial condition
    :math:`\mathbf{z}_0` nor the information of the solution to the next iteration on the left-hand side :math:`\mathbf{z}^{k+1}`. That means
    quadrature is still applied to all parts of the problem but due to the embedding information just mentioned are not used.
    """

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

        u = self.dtype_u(u0)

        # start newton iteration
        n = 0
        res = 99
        while n < self.newton_maxiter:
            u1, u2, u3 = u.diff[0], u.diff[1], u.diff[2]

            f = self.eval_f(u, t)

            # form the function g(u), such that the solution to the nonlinear problem is a root of g
            g = np.array(
                [
                    u1 - factor * f.diff[0] - rhs.diff[0],
                    u2 - factor * f.diff[1] - rhs.diff[1],
                    u3 - factor * f.diff[2] - rhs.diff[2],
                    -factor * f.alg[0] - rhs.alg[0],
                ]
            )

            # if g is close to 0, then we are done
            res = np.linalg.norm(g, np.inf)
            if res < self.newton_tol:
                break

            # assemble dg
            dg = np.array(
                [
                    [1 - factor, 0, factor, -factor],
                    [0, 1 + factor * 1e4, 0, 0],
                    [-factor, 0, 1, 0],
                    [-factor, -factor, 0, -factor],
                ]
            )

            # newton update: u1 = u0 - g/dg
            dx = np.linalg.solve(dg, g)

            u.diff[0] -= dx[0]
            u.diff[1] -= dx[1]
            u.diff[2] -= dx[2]
            u.alg[0] -= dx[3]

            n += 1
            self.work_counters['newton']()

        if np.isnan(res) and self.stop_at_nan:
            raise ProblemError('Newton got nan after %i iterations, aborting...' % n)
        elif np.isnan(res):
            self.logger.warning('Newton got nan after %i iterations...' % n)

        if n == self.newton_maxiter:
            msg = 'Newton did not converge after %i iterations, error is %s' % (n, res)
            if self.stop_at_maxiter:
                raise ProblemError(msg)
            else:
                self.logger.warning(msg)

        me = self.dtype_u(self.init)
        me[:] = u[:]
        return me
