r"""
1D Allen-Cahn equation – finite-difference IMEX formulation
============================================================

We solve

.. math::

    \frac{\partial u}{\partial t}
    = \underbrace{\frac{\partial^2 u}{\partial x^2}}_{f_\text{impl}}
      \underbrace{- \frac{2}{\varepsilon^2}\,u(1-u)(1-2u)
                  - 6\,d_w\,u(1-u)}_{f_\text{expl}},
    \quad x \in [a, b],

with time-dependent Dirichlet boundary conditions taken from the exact
travelling-wave solution

.. math::

    u(x, t)
    = \tfrac{1}{2}\!\left(1 + \tanh\!\left(
          \frac{x - v\,t}{\sqrt{2}\,\varepsilon}\right)\right),
    \quad v = 3\sqrt{2}\,\varepsilon\,d_w.

**IMEX splitting**

* *Implicit* part  :math:`f_\text{impl}(u) = u_{xx}` – handled by a direct
  sparse linear solve at each collocation node.
* *Explicit* part  :math:`f_\text{expl}(u)` – the nonlinear reaction term,
  evaluated explicitly.

**Spatial discretisation**

Second-order centred finite differences on a uniform grid of *nvars*
interior points.  The time-dependent boundary values enter the interior
equations only at the first and last node, so they are collected in a
boundary-correction vector :math:`b_\text{bc}(t)`:

.. math::

    (A\,u)_i + (b_\text{bc})_i
    \approx \frac{\partial^2 u}{\partial x^2}(x_i, t),

where :math:`A` is the :math:`n \times n` tridiagonal matrix with
:math:`(1,\,-2,\,1)/\Delta x^2` on the three diagonals, and

.. math::

    (b_\text{bc})_i =
    \begin{cases}
        u_\text{left}(t) / \Delta x^2 & i = 0, \\
        u_\text{right}(t) / \Delta x^2 & i = n-1, \\
        0 & \text{otherwise.}
    \end{cases}
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

from pySDC.core.problem import Problem, WorkCounter
from pySDC.implementations.datatype_classes.mesh import mesh, imex_mesh


class allencahn_1d_imex(Problem):
    r"""
    Problem class for the 1D Allen-Cahn equation with driving force, using
    second-order centred finite differences and an IMEX (semi-implicit)
    splitting.

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
        Finite-difference Laplacian for the interior points.
    """

    dtype_u = mesh
    dtype_f = imex_mesh

    def __init__(self, nvars=127, eps=0.04, dw=-0.04, interval=(-0.5, 0.5)):
        super().__init__((nvars, None, np.dtype('float64')))
        self._makeAttributeAndRegister('nvars', 'eps', 'dw', 'interval', localVars=locals(), readOnly=True)

        self.dx = (interval[1] - interval[0]) / (nvars + 1)
        self.xvalues = np.linspace(interval[0] + self.dx, interval[1] - self.dx, nvars)

        # Build the tridiagonal FD Laplacian (interior points only).
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
        """Evaluate the exact travelling-wave profile at positions *x* and time *t*."""
        v = 3.0 * np.sqrt(2.0) * self.eps * self.dw
        return 0.5 * (1.0 + np.tanh((x - v * t) / (np.sqrt(2.0) * self.eps)))

    def _bc_vector(self, t):
        r"""
        Boundary-correction vector :math:`b_\text{bc}(t)`.

        Returns the length-*nvars* vector such that
        ``A @ u + _bc_vector(t)`` approximates :math:`u_{xx}` at the
        interior grid points with time-dependent Dirichlet values at both
        domain boundaries.
        """
        bc = np.zeros(self.nvars)
        bc[0] = self._travelling_wave(self.interval[0], t) / self.dx**2
        bc[-1] = self._travelling_wave(self.interval[1], t) / self.dx**2
        return bc

    # ------------------------------------------------------------------
    # Required pySDC interface
    # ------------------------------------------------------------------

    def eval_f(self, u, t):
        r"""
        Evaluate the right-hand side, split into implicit and explicit parts.

        Parameters
        ----------
        u : dtype_u
            Current solution on the interior grid.
        t : float
            Current time.

        Returns
        -------
        f : imex_mesh
            ``f.impl`` = :math:`u_{xx}` (Laplacian with BCs),
            ``f.expl`` = reaction term.
        """
        f = self.dtype_f(self.init)
        f.impl[:] = self.A.dot(u) + self._bc_vector(t)
        f.expl[:] = -2.0 / self.eps**2 * u * (1.0 - u) * (1.0 - 2.0 * u) - 6.0 * self.dw * u * (1.0 - u)
        self.work_counters['rhs']()
        return f

    def solve_system(self, rhs, factor, u0, t):
        r"""
        Solve the linear system arising from the implicit Euler step:

        .. math::

            \bigl(I - \text{factor}\cdot A\bigr)\,u
            = \text{rhs} + \text{factor}\cdot b_\text{bc}(t).

        Parameters
        ----------
        rhs : dtype_u
            Right-hand side supplied by the sweeper.
        factor : float
            Implicit step size (``dt * QI[m, m]``).
        u0 : dtype_u
            Initial guess (not used; direct solver).
        t : float
            Current time (needed for the Dirichlet boundary values).

        Returns
        -------
        me : dtype_u
            Solution of the linear system.
        """
        me = self.dtype_u(self.init)
        system_rhs = np.asarray(rhs) + factor * self._bc_vector(t)
        me[:] = spsolve(sp.eye(self.nvars, format='csc') - factor * self.A, system_rhs)
        return me

    def u_exact(self, t):
        r"""
        Exact travelling-wave solution on the interior grid at time *t*.

        Parameters
        ----------
        t : float
            Time at which to evaluate the exact solution.

        Returns
        -------
        me : dtype_u
            Exact solution values at the interior grid points.
        """
        me = self.dtype_u(self.init, val=0.0)
        me[:] = self._travelling_wave(self.xvalues, t)
        return me
