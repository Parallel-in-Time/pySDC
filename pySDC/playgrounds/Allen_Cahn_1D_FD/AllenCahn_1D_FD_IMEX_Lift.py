r"""
1D Allen-Cahn equation – finite-difference IMEX formulation with boundary lifting
==================================================================================

This module provides :class:`allencahn_1d_imex_lift`, a boundary-lifted
variant of the standard FD/IMEX formulation in
:mod:`AllenCahn_1D_FD_IMEX`.

**Why boundary lifting?**

When time-dependent Dirichlet BCs are handled by adding a correction vector
:math:`b_\text{bc}(t)` to the implicit right-hand side (as in
:class:`~AllenCahn_1D_FD_IMEX.allencahn_1d_imex`), the implicit operator
:math:`f_\text{impl}` acquires an explicit time dependence.  This breaks
the standard SDC convergence argument (which assumes an autonomous implicit
operator) and leads to *order reduction*: the effective temporal order
stalls below the expected value :math:`K` after :math:`K` sweeps.

**Boundary lifting** eliminates the issue by introducing the change of
variable

.. math::

    v(x, t) = u(x, t) - E(x, t),

where :math:`E` is a *lift function* that satisfies the time-dependent
Dirichlet BCs at every time :math:`t`.  We choose a linear interpolant in
space:

.. math::

    E(x, t) = u_\text{left}(t)
              + \bigl[u_\text{right}(t) - u_\text{left}(t)\bigr]
                \frac{x - a}{b - a},

where :math:`u_\text{left/right}(t)` are the exact travelling-wave values
at the domain boundaries :math:`x = a` and :math:`x = b`.

The transformed variable :math:`v` satisfies *homogeneous* Dirichlet BCs
(:math:`v = 0` at :math:`x = a` and :math:`x = b`) and evolves according
to

.. math::

    \frac{\partial v}{\partial t}
    = A\,v
      \underbrace{
        + f_\text{expl}(v + E(t)) - \dot{E}(t)
      }_{f_\text{expl, lifted}},

where :math:`A` is the same tridiagonal FD Laplacian and :math:`\dot{E}(t)`
is the pointwise time derivative of the lift.  The key property is that
:math:`A\,v` already contains the correct discrete Laplacian of :math:`u`
*without* any extra :math:`b_\text{bc}` correction, because the linear lift
has zero second derivative (both continuous and discrete).

The IMEX split for the lifted problem is therefore:

* *Implicit* part  :math:`f_\text{impl}(v) = A\,v`
  – purely autonomous linear operator, **no BC correction needed in**
  ``solve_system``.
* *Explicit* part
  :math:`f_\text{expl}(v, t) = f_\text{expl, reaction}(v + E(t)) - \dot{E}(t)`
  – nonlinear reaction applied to the physical variable :math:`u = v + E(t)`,
  plus a source term that accounts for the moving lift.
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

from pySDC.core.problem import Problem, WorkCounter
from pySDC.implementations.datatype_classes.mesh import mesh, imex_mesh


class allencahn_1d_imex_lift(Problem):
    r"""
    Boundary-lifted problem class for the 1D Allen-Cahn equation.

    The *state variable* is

    .. math::

        v(x, t) = u(x, t) - E(x, t),

    where :math:`u` is the physical solution and :math:`E(x,t)` is a linear
    lift that exactly satisfies the time-dependent Dirichlet BCs.  The
    variable :math:`v` satisfies *homogeneous* BCs, which allows the SDC
    sweeper to achieve its full theoretical convergence order.

    To recover the physical solution at the end of a run, call
    :meth:`lift` and add its result to the returned state vector.

    Parameters
    ----------
    nvars : int
        Number of interior grid points (default 127).
    eps : float
        Interface-width parameter :math:`\varepsilon` (default 0.04).
    dw : float
        Driving-force parameter (default -0.04).
    interval : tuple of float
        Spatial domain :math:`[a, b]` (default ``(-0.5, 0.5)``).

    Attributes
    ----------
    dx : float
        Uniform grid spacing.
    xvalues : numpy.ndarray, shape (nvars,)
        Coordinates of the interior grid points.
    A : scipy.sparse.csc_matrix, shape (nvars, nvars)
        Finite-difference Laplacian for the interior points (with zero BCs).
    """

    dtype_u = mesh
    dtype_f = imex_mesh

    def __init__(self, nvars=127, eps=0.04, dw=-0.04, interval=(-0.5, 0.5)):
        super().__init__((nvars, None, np.dtype('float64')))
        self._makeAttributeAndRegister('nvars', 'eps', 'dw', 'interval', localVars=locals(), readOnly=True)

        self.dx = (interval[1] - interval[0]) / (nvars + 1)
        self.xvalues = np.linspace(interval[0] + self.dx, interval[1] - self.dx, nvars)

        # Tridiagonal FD Laplacian for the interior (homogeneous BCs).
        diag_main = np.full(nvars, -2.0)
        diag_off = np.ones(nvars - 1)
        self.A = (
            sp.diags([diag_off, diag_main, diag_off], offsets=[-1, 0, 1], shape=(nvars, nvars), format='csc')
            / self.dx**2
        )

        self.work_counters['rhs'] = WorkCounter()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _travelling_wave(self, x, t):
        """Exact travelling-wave profile at positions *x* and time *t*.

        Parameters
        ----------
        x : float or numpy.ndarray
            Spatial position(s).
        t : float
            Time.

        Returns
        -------
        numpy.ndarray or float
            Travelling-wave values.
        """
        v_wave = 3.0 * np.sqrt(2.0) * self.eps * self.dw
        return 0.5 * (1.0 + np.tanh((x - v_wave * t) / (np.sqrt(2.0) * self.eps)))

    def _travelling_wave_dt(self, x, t):
        r"""
        Time derivative :math:`\partial u / \partial t` of the exact wave.

        .. math::

            \frac{\partial u}{\partial t}(x, t)
            = -\frac{v}{2\sqrt{2}\,\varepsilon}
              \operatorname{sech}^2\!\left(
                \frac{x - v\,t}{\sqrt{2}\,\varepsilon}
              \right),
            \quad v = 3\sqrt{2}\,\varepsilon\,d_w.

        Parameters
        ----------
        x : float or numpy.ndarray
            Spatial position(s).
        t : float
            Time.

        Returns
        -------
        numpy.ndarray or float
            Time-derivative values.
        """
        v_wave = 3.0 * np.sqrt(2.0) * self.eps * self.dw
        xi = (x - v_wave * t) / (np.sqrt(2.0) * self.eps)
        return -v_wave / (np.sqrt(2.0) * self.eps) * 0.5 / np.cosh(xi) ** 2

    def lift(self, t):
        r"""
        Lift vector :math:`E(x_i, t)` at interior grid points.

        Linear interpolation between the exact boundary values:

        .. math::

            E(x_i, t) = u_\text{left}(t)
                + \bigl[u_\text{right}(t) - u_\text{left}(t)\bigr]
                  \frac{x_i - a}{b - a}.

        Parameters
        ----------
        t : float
            Current time.

        Returns
        -------
        E : numpy.ndarray, shape (nvars,)
        """
        a, b = self.interval
        u_left = self._travelling_wave(a, t)
        u_right = self._travelling_wave(b, t)
        return u_left + (u_right - u_left) * (self.xvalues - a) / (b - a)

    def _dlift_dt(self, t):
        r"""
        Time derivative :math:`\dot{E}(x_i, t)` of the lift at interior points.

        .. math::

            \dot{E}(x_i, t)
            = \dot{u}_\text{left}(t)
              + \bigl[\dot{u}_\text{right}(t) - \dot{u}_\text{left}(t)\bigr]
                \frac{x_i - a}{b - a}.
        """
        a, b = self.interval
        dudt_left = self._travelling_wave_dt(a, t)
        dudt_right = self._travelling_wave_dt(b, t)
        return dudt_left + (dudt_right - dudt_left) * (self.xvalues - a) / (b - a)

    # ------------------------------------------------------------------
    # Required pySDC interface
    # ------------------------------------------------------------------

    def eval_f(self, v, t):
        r"""
        Evaluate the right-hand side for the lifted variable :math:`v`.

        Parameters
        ----------
        v : dtype_u
            Current lifted solution (interior grid).
        t : float
            Current time.

        Returns
        -------
        f : imex_mesh
            ``f.impl`` = :math:`A\,v` (Laplacian with homogeneous BCs, **no**
            boundary-correction vector needed),
            ``f.expl`` = :math:`f_\text{expl,reaction}(v + E(t)) - \dot{E}(t)`.
        """
        E = self.lift(t)
        u = np.asarray(v) + E  # recover the physical solution

        f = self.dtype_f(self.init)
        # Implicit part: pure Laplacian of v with homogeneous BCs.
        f.impl[:] = self.A.dot(v)
        # Explicit part: nonlinear reaction of u, minus the lift time derivative.
        reaction = -2.0 / self.eps**2 * u * (1.0 - u) * (1.0 - 2.0 * u) - 6.0 * self.dw * u * (1.0 - u)
        f.expl[:] = reaction - self._dlift_dt(t)
        self.work_counters['rhs']()
        return f

    def solve_system(self, rhs, factor, u0, t):
        r"""
        Solve the linear system for :math:`v` (homogeneous BCs, no correction).

        .. math::

            \bigl(I - \text{factor}\cdot A\bigr)\,v = \text{rhs}.

        No boundary-correction term is needed because the implicit operator
        :math:`A` acts on :math:`v`, which satisfies zero Dirichlet BCs.

        Parameters
        ----------
        rhs : dtype_u
            Right-hand side from the sweeper.
        factor : float
            Implicit step size.
        u0 : dtype_u
            Initial guess (not used; direct solver).
        t : float
            Current time (not used; kept for interface compatibility).

        Returns
        -------
        me : dtype_u
            Solution :math:`v`.
        """
        me = self.dtype_u(self.init)
        me[:] = spsolve(sp.eye(self.nvars, format='csc') - factor * self.A, np.asarray(rhs))
        return me

    def u_exact(self, t):
        r"""
        Exact solution for the lifted variable at time *t*:

        .. math::

            v_\text{exact}(x_i, t)
            = u_\text{exact}(x_i, t) - E(x_i, t).

        To recover the physical solution, add :meth:`lift(t)`.

        Parameters
        ----------
        t : float
            Time at which to evaluate the exact solution.

        Returns
        -------
        me : dtype_u
            Exact lifted solution at interior grid points.
        """
        me = self.dtype_u(self.init, val=0.0)
        me[:] = self._travelling_wave(self.xvalues, t) - self.lift(t)
        return me
