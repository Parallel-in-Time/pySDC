r"""
Manufactured-solution problem classes for the 1D viscous Burgers playground
============================================================================

Two self-contained problem classes on the domain :math:`[0, 1]`, each
implementing the viscous Burgers equation with a source term that enforces a
prescribed smooth exact solution:

.. math::

    u_t = \nu\,u_{xx} - u\,u_x + g(x, t),

where :math:`g(x,t)` is chosen so that :math:`u_\text{ex}` is the exact
solution.  The two classes differ only in their boundary treatment:

* :class:`burgers_1d_hom` – homogeneous Dirichlet BCs; no
  boundary-correction vector :math:`b_\text{bc}` required.
  The implicit operator :math:`f_\text{impl} = \nu A\,u` is autonomous.
* :class:`burgers_1d_inhom` – time-dependent Dirichlet BCs handled
  via the standard :math:`b_\text{bc}(t)` correction in
  :math:`f_\text{impl}`.

Comparing the two cases under fully-converged IMEX-SDC shows that
the homogeneous case achieves the full collocation order :math:`2M-1`, while
including a time-dependent :math:`b_\text{bc}(t)` in :math:`f_\text{impl}`
reduces the collocation order (order reduction).

**Spatial discretisation**

Both classes use a **fourth-order finite-difference** Laplacian and a
**fourth-order finite-difference** first-derivative operator on :math:`[0,1]`,
so that spatial errors are :math:`O(\Delta x^4)` and do not dominate the
temporal error for the grid resolutions used.

Classes
-------
burgers_1d_hom
    :math:`u_\text{ex}(x,t) = 0.1\sin(\pi x)\cos(t)`,
    homogeneous BCs :math:`u|_\partial = 0`.

burgers_1d_inhom
    :math:`u_\text{ex}(x,t) = 0.5 + 0.1\sin(\pi x)\cos(t) + 0.1\,x\sin(t)`,
    time-dependent BCs :math:`u(0,t)=0.5`,
    :math:`u(1,t)=0.5+0.1\sin(t)`.  Standard :math:`b_\text{bc}(t)` correction.
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

from pySDC.core.problem import Problem, WorkCounter
from pySDC.implementations.datatype_classes.mesh import mesh, imex_mesh


# ---------------------------------------------------------------------------
# Shared base class
# ---------------------------------------------------------------------------

class _Burgers1D_Base(Problem):
    r"""
    Shared setup for all problem classes.

    Builds a fourth-order FD Laplacian on :math:`[0,1]` with *zero* Dirichlet
    BCs (the boundary correction, if any, is handled by the subclass), and
    provides a fourth-order FD first-derivative helper.

    **Laplacian** (second derivative)

    Interior rows :math:`k = 2, \ldots, n-3` use the standard centred
    fourth-order stencil

    .. math::

        u_{xx}(x_k) \approx
        \frac{-u_{k-2} + 16u_{k-1} - 30u_k + 16u_{k+1} - u_{k+2}}{12\,\Delta x^2}.

    Rows :math:`k=1` and :math:`k=n-2` use the same centred stencil with the
    known Dirichlet boundary value substituted, contributing to
    :math:`b_\text{bc}` in the inhomogeneous subclass.

    Rows :math:`k=0` and :math:`k=n-1` use a 6-point one-sided stencil:

    .. math::

        u_{xx}(x_0) \approx
        \frac{10u_L - 15u_0 - 4u_1 + 14u_2 - 6u_3 + u_4}{12\,\Delta x^2}.

    **First derivative**

    The static method :meth:`_compute_ux` computes :math:`u_x` at interior
    grid points using a fourth-order finite-difference stencil:

    * :math:`k=0`: 4th-order one-sided forward stencil.
    * :math:`k=1`: 4th-order centred stencil using :math:`u_L` as left ghost.
    * :math:`k=2,\ldots,n-3`: standard 4th-order centred stencil.
    * :math:`k=n-2`: 4th-order centred stencil using :math:`u_R` as right ghost.
    * :math:`k=n-1`: 4th-order one-sided backward stencil.

    Parameters
    ----------
    nvars : int
        Number of interior grid points (default 127; must be ≥ 5).
    nu : float
        Kinematic viscosity :math:`\nu` (default 0.1).

    Attributes
    ----------
    dx : float
        Grid spacing :math:`1/(n+1)`.
    xvalues : numpy.ndarray
        Interior grid point coordinates.
    A : scipy.sparse.csc_matrix
        Fourth-order FD Laplacian (zero Dirichlet BCs, autonomous part only).
    """

    dtype_u = mesh
    dtype_f = imex_mesh

    def __init__(self, nvars=127, nu=0.1):
        if nvars < 5:
            raise ValueError(f"nvars must be >= 5 for the 4th-order FD operators; got {nvars}")
        super().__init__((nvars, None, np.dtype('float64')))
        self._makeAttributeAndRegister('nvars', 'nu', localVars=locals(), readOnly=True)

        self.dx = 1.0 / (nvars + 1)
        self.xvalues = np.linspace(self.dx, 1.0 - self.dx, nvars)
        self.A = self._build_laplacian(nvars, self.dx)
        self.work_counters['rhs'] = WorkCounter()

    @staticmethod
    def _build_laplacian(n, dx):
        r"""
        Assemble the fourth-order FD Laplacian matrix (n × n, zero Dirichlet BCs).

        Parameters
        ----------
        n : int
            Number of interior grid points (>= 5).
        dx : float
            Grid spacing.

        Returns
        -------
        scipy.sparse.csc_matrix
        """
        inv12dx2 = 1.0 / (12.0 * dx**2)
        A = sp.lil_matrix((n, n))

        # Row 0: 6-point one-sided stencil (4th-order, no ghost point).
        # u_xx(x_0) ~ (10*u_L - 15*u[0] - 4*u[1] + 14*u[2] - 6*u[3] + u[4]) / (12*dx^2)
        # Matrix part (u_L=0 for zero-BC; non-zero contribution goes to b_bc):
        A[0, 0] = -15
        A[0, 1] = -4
        A[0, 2] = 14
        A[0, 3] = -6
        A[0, 4] = 1

        # Row 1: standard 4th-order centred, uses u[-1]=u_L as b_bc correction.
        # u_xx(x_1) ~ (-u_L + 16*u[0] - 30*u[1] + 16*u[2] - u[3]) / (12*dx^2)
        A[1, 0] = 16
        A[1, 1] = -30
        A[1, 2] = 16
        A[1, 3] = -1

        # Rows 2,...,n-3: standard centred 4th-order stencil.
        for k in range(2, n - 2):
            A[k, k - 2] = -1
            A[k, k - 1] = 16
            A[k, k] = -30
            A[k, k + 1] = 16
            A[k, k + 2] = -1

        # Row n-2: standard 4th-order centred, uses u[n]=u_R as b_bc correction.
        # u_xx(x_{n-2}) ~ (-u[n-4]+16*u[n-3]-30*u[n-2]+16*u[n-1]-u_R)/(12*dx^2)
        A[n - 2, n - 4] = -1
        A[n - 2, n - 3] = 16
        A[n - 2, n - 2] = -30
        A[n - 2, n - 1] = 16

        # Row n-1: mirror of row 0 (6-point one-sided stencil).
        # u_xx(x_{n-1}) ~ (u[n-5]-6*u[n-4]+14*u[n-3]-4*u[n-2]-15*u[n-1]+10*u_R)/(12*dx^2)
        A[n - 1, n - 5] = 1
        A[n - 1, n - 4] = -6
        A[n - 1, n - 3] = 14
        A[n - 1, n - 2] = -4
        A[n - 1, n - 1] = -15

        return (A * inv12dx2).tocsc()

    @staticmethod
    def _compute_ux(u, u_L, u_R, dx):
        r"""
        Fourth-order FD first derivative at interior grid points.

        The stencil is:

        * :math:`k=0`: 4th-order one-sided forward
          :math:`(-25u_0+48u_1-36u_2+16u_3-3u_4)/(12\Delta x)`.
        * :math:`k=1`: centred, left ghost :math:`u_L`
          :math:`(u_L-8u_0+8u_2-u_3)/(12\Delta x)`.
        * :math:`k=2,\ldots,n-3`: standard centred
          :math:`(u_{k-2}-8u_{k-1}+8u_{k+1}-u_{k+2})/(12\Delta x)`.
        * :math:`k=n-2`: centred, right ghost :math:`u_R`
          :math:`(u_{n-4}-8u_{n-3}+8u_{n-1}-u_R)/(12\Delta x)`.
        * :math:`k=n-1`: 4th-order one-sided backward
          :math:`(3u_{n-5}-16u_{n-4}+36u_{n-3}-48u_{n-2}+25u_{n-1})/(12\Delta x)`.

        Parameters
        ----------
        u : numpy.ndarray
            Interior solution values (length n, n >= 5).
        u_L : float
            Left boundary value.
        u_R : float
            Right boundary value.
        dx : float
            Grid spacing.

        Returns
        -------
        ux : numpy.ndarray
            First derivative at interior grid points.
        """
        n = len(u)
        ux = np.empty(n)
        inv12dx = 1.0 / (12.0 * dx)

        # k=0: 4th-order one-sided forward stencil (no ghost needed)
        ux[0] = (-25 * u[0] + 48 * u[1] - 36 * u[2] + 16 * u[3] - 3 * u[4]) * inv12dx

        # k=1: centred, uses u_L as left ghost value
        ux[1] = (u_L - 8 * u[0] + 8 * u[2] - u[3]) * inv12dx

        # k=2,...,n-3: standard 4th-order centred (vectorised)
        ux[2:n - 2] = (u[:n - 4] - 8 * u[1:n - 3] + 8 * u[3:n - 1] - u[4:]) * inv12dx

        # k=n-2: centred, uses u_R as right ghost value
        ux[n - 2] = (u[n - 4] - 8 * u[n - 3] + 8 * u[n - 1] - u_R) * inv12dx

        # k=n-1: 4th-order one-sided backward stencil (no ghost needed)
        ux[n - 1] = (3 * u[n - 5] - 16 * u[n - 4] + 36 * u[n - 3] - 48 * u[n - 2] + 25 * u[n - 1]) * inv12dx

        return ux


# ---------------------------------------------------------------------------
# Case 1: Homogeneous BCs
# ---------------------------------------------------------------------------

class burgers_1d_hom(_Burgers1D_Base):
    r"""
    Viscous Burgers problem with **homogeneous** Dirichlet BCs.

    **Exact solution**

    .. math::

        u_\text{ex}(x, t) = 0.1\,\sin(\pi x)\,\cos(t), \quad x \in [0, 1].

    **Boundary conditions**

    :math:`u(0, t) = 0`,  :math:`u(1, t) = 0`  (homogeneous, time-independent).

    **Forcing term**

    The source term :math:`g(x,t)` is chosen so that :math:`u_\text{ex}` satisfies

    .. math::

        u_t = \nu\,u_{xx} - u\,u_x + g(x, t)

    exactly:

    .. math::

        g = u_t^\text{ex} - \nu\,u_{xx}^\text{ex} + u_\text{ex}\,u_x^\text{ex}.

    **IMEX split**

    * :math:`f_\text{impl} = \nu A\,u` – autonomous Laplacian, no
      boundary-correction vector.
    * :math:`f_\text{expl} = -u\,u_x + g(x, t)`.
    """

    def _forcing(self, t):
        r"""
        Manufactured forcing :math:`g = u_t - \nu u_{xx} + u\,u_x`.

        Parameters
        ----------
        t : float

        Returns
        -------
        numpy.ndarray
        """
        x = self.xvalues
        u_ex = 0.1 * np.sin(np.pi * x) * np.cos(t)
        u_t = -0.1 * np.sin(np.pi * x) * np.sin(t)
        u_xx = -0.1 * np.pi**2 * np.sin(np.pi * x) * np.cos(t)
        u_x = 0.1 * np.pi * np.cos(np.pi * x) * np.cos(t)
        return u_t - self.nu * u_xx + u_ex * u_x

    def eval_f(self, u, t):
        r"""
        Evaluate :math:`f_\text{impl}` and :math:`f_\text{expl}`.

        Parameters
        ----------
        u : dtype_u
            Current solution.
        t : float
            Current time.

        Returns
        -------
        f : imex_mesh
            ``f.impl = nu * A * u``, ``f.expl = -u * u_x + g(t)``.
        """
        f = self.dtype_f(self.init)
        u_arr = np.asarray(u)
        f.impl[:] = self.nu * self.A.dot(u_arr)
        u_x = self._compute_ux(u_arr, 0.0, 0.0, self.dx)
        f.expl[:] = -u_arr * u_x + self._forcing(t)
        self.work_counters['rhs']()
        return f

    def solve_system(self, rhs, factor, u0, t):
        r"""
        Solve :math:`(I - \text{factor}\cdot\nu A)\,u = \text{rhs}`.

        No boundary-correction term (homogeneous BCs).

        Parameters
        ----------
        rhs : dtype_u
        factor : float
        u0 : dtype_u
        t : float

        Returns
        -------
        me : dtype_u
        """
        me = self.dtype_u(self.init)
        me[:] = spsolve(
            sp.eye(self.nvars, format='csc') - factor * self.nu * self.A,
            np.asarray(rhs),
        )
        return me

    def u_exact(self, t):
        r"""
        Exact solution :math:`0.1\sin(\pi x)\cos(t)` at interior grid points.

        Parameters
        ----------
        t : float

        Returns
        -------
        me : dtype_u
        """
        me = self.dtype_u(self.init, val=0.0)
        me[:] = 0.1 * np.sin(np.pi * self.xvalues) * np.cos(t)
        return me


# ---------------------------------------------------------------------------
# Case 2: Inhomogeneous BCs, standard b_bc correction
# ---------------------------------------------------------------------------

class burgers_1d_inhom(_Burgers1D_Base):
    r"""
    Viscous Burgers problem with **time-dependent** Dirichlet BCs,
    handled by the standard boundary-correction vector :math:`b_\text{bc}(t)`.

    **Exact solution**

    .. math::

        u_\text{ex}(x, t) = 0.5 + 0.1\,\sin(\pi x)\,\cos(t) + 0.1\,x\,\sin(t),
        \quad x \in [0, 1].

    **Boundary conditions**

    :math:`u(0, t) = 0.5` (constant),
    :math:`u(1, t) = 0.5 + 0.1\sin(t)` (time-dependent).

    **Forcing term**

    .. math::

        g = u_t^\text{ex} - \nu\,u_{xx}^\text{ex} + u_\text{ex}\,u_x^\text{ex}.

    **IMEX split**

    * :math:`f_\text{impl} = \nu A\,u + \nu\,b_\text{bc}(t)`
      – includes the time-dependent boundary-correction vector.
    * :math:`f_\text{expl} = -u\,u_x + g(x, t)`.

    The time-dependent right BC :math:`u(1,t) = 0.5 + 0.1\sin(t)` makes
    :math:`b_\text{bc}(t)` explicitly time-dependent, which causes order
    reduction compared to the homogeneous case.

    **Boundary-correction vector**

    For the 4th-order FD Laplacian, the non-zero entries of the unscaled
    :math:`b_\text{bc}` come from substituting the Dirichlet values into the
    stencil rows that touch the domain boundary:

    * Row 0 (6-point one-sided): :math:`10\,u_L/(12\,\Delta x^2)`.
    * Row 1 (centred, needs :math:`u_L` as ghost): :math:`-u_L/(12\,\Delta x^2)`.
    * Row :math:`n-2` (centred, needs :math:`u_R` as ghost): :math:`-u_R/(12\,\Delta x^2)`.
    * Row :math:`n-1` (6-point one-sided): :math:`10\,u_R/(12\,\Delta x^2)`.

    The vector returned by :meth:`_bc_vector` is scaled by :math:`\nu`.
    """

    def _bc_values(self, t):
        r"""
        Dirichlet boundary values :math:`(u_L, u_R)` at time *t*.

        Parameters
        ----------
        t : float

        Returns
        -------
        u_L : float
            Left boundary value :math:`0.5`.
        u_R : float
            Right boundary value :math:`0.5 + 0.1\sin(t)`.
        """
        return 0.5, 0.5 + 0.1 * np.sin(t)

    def _bc_vector(self, t):
        r"""
        Boundary-correction vector :math:`\nu\,b_\text{bc}(t)`.

        Parameters
        ----------
        t : float

        Returns
        -------
        numpy.ndarray
        """
        n = self.nvars
        inv12dx2 = 1.0 / (12.0 * self.dx**2)
        u_L, u_R = self._bc_values(t)
        bc = np.zeros(n)
        bc[0] = 10 * u_L * inv12dx2
        bc[1] = -u_L * inv12dx2
        bc[n - 2] = -u_R * inv12dx2
        bc[n - 1] = 10 * u_R * inv12dx2
        return self.nu * bc

    def _forcing(self, t):
        r"""
        Manufactured forcing
        :math:`g = u_t - \nu u_{xx} + u_\text{ex}\,u_x^\text{ex}`.

        Parameters
        ----------
        t : float

        Returns
        -------
        numpy.ndarray
        """
        x = self.xvalues
        u_ex = 0.5 + 0.1 * np.sin(np.pi * x) * np.cos(t) + 0.1 * x * np.sin(t)
        u_t = -0.1 * np.sin(np.pi * x) * np.sin(t) + 0.1 * x * np.cos(t)
        u_xx = -0.1 * np.pi**2 * np.sin(np.pi * x) * np.cos(t)
        u_x = 0.1 * np.pi * np.cos(np.pi * x) * np.cos(t) + 0.1 * np.sin(t)
        return u_t - self.nu * u_xx + u_ex * u_x

    def eval_f(self, u, t):
        r"""
        Evaluate :math:`f_\text{impl} = \nu A u + \nu b_\text{bc}(t)` and
        :math:`f_\text{expl} = -u\,u_x + g(t)`.

        Parameters
        ----------
        u : dtype_u
        t : float

        Returns
        -------
        f : imex_mesh
        """
        f = self.dtype_f(self.init)
        u_arr = np.asarray(u)
        f.impl[:] = self.nu * self.A.dot(u_arr) + self._bc_vector(t)
        u_L, u_R = self._bc_values(t)
        u_x = self._compute_ux(u_arr, u_L, u_R, self.dx)
        f.expl[:] = -u_arr * u_x + self._forcing(t)
        self.work_counters['rhs']()
        return f

    def solve_system(self, rhs, factor, u0, t):
        r"""
        Solve
        :math:`(I - \text{factor}\cdot\nu A)\,u = \text{rhs} + \text{factor}\cdot\nu b_\text{bc}(t)`.

        Parameters
        ----------
        rhs : dtype_u
        factor : float
        u0 : dtype_u
        t : float

        Returns
        -------
        me : dtype_u
        """
        me = self.dtype_u(self.init)
        system_rhs = np.asarray(rhs) + factor * self._bc_vector(t)
        me[:] = spsolve(
            sp.eye(self.nvars, format='csc') - factor * self.nu * self.A,
            system_rhs,
        )
        return me

    def u_exact(self, t):
        r"""
        Exact solution
        :math:`0.5 + 0.1\sin(\pi x)\cos(t) + 0.1\,x\sin(t)`.

        Parameters
        ----------
        t : float

        Returns
        -------
        me : dtype_u
        """
        me = self.dtype_u(self.init, val=0.0)
        x = self.xvalues
        me[:] = 0.5 + 0.1 * np.sin(np.pi * x) * np.cos(t) + 0.1 * x * np.sin(t)
        return me
