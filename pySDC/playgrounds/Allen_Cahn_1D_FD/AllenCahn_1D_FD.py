r"""
Manufactured-solution problem classes for the 1D Allen-Cahn playground
==============================================================================

Three self-contained problem classes on the domain :math:`[0, 1]`, each
implementing the Allen-Cahn equation with a source term that enforces a
prescribed smooth exact solution:

.. math::

    u_t = u_{xx} + R(u) + g(x, t),

where

.. math::

    R(u) = -\frac{2}{\varepsilon^2}\,u(1-u)(1-2u) - 6\,d_w\,u(1-u)

and :math:`g(x,t)` is chosen so that :math:`u_\text{ex}` is the exact
solution.  The three classes differ only in their boundary treatment:

* :class:`allencahn_1d_hom` – homogeneous Dirichlet BCs; no
  boundary-correction vector :math:`b_\text{bc}` required.
  The implicit operator :math:`f_\text{impl} = A u` is autonomous.
* :class:`allencahn_1d_inhom` – time-dependent Dirichlet BCs handled
  via the standard :math:`b_\text{bc}(t)` correction in
  :math:`f_\text{impl}`.
* :class:`allencahn_1d_inhom_lift` – same time-dependent BCs treated
  by boundary lifting (:math:`v = u - E`); the implicit operator
  :math:`f_\text{impl} = A v` is again autonomous.

Comparing the three cases under fully-converged IMEX-SDC shows that
including a time-dependent :math:`b_\text{bc}(t)` in :math:`f_\text{impl}`
reduces the collocation order, while boundary lifting restores it.

Classes
-------
allencahn_1d_hom
    :math:`u_\text{ex}(x,t) = \sin(\pi x)\cos(t)`,
    homogeneous BCs :math:`u|_\partial = 0`.

allencahn_1d_inhom
    :math:`u_\text{ex}(x,t) = \cos(\pi x)\cos(t)`,
    time-dependent BCs :math:`u(0,t)=\cos(t)`,
    :math:`u(1,t)=-\cos(t)`.  Standard :math:`b_\text{bc}(t)` correction.

allencahn_1d_inhom_lift
    Same exact solution as :class:`allencahn_1d_inhom` but reformulated
    with boundary lifting.  Lift :math:`E(x,t) = (1-2x)\cos(t)`, state
    variable :math:`v = u - E` satisfies homogeneous BCs.
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

from pySDC.core.problem import Problem, WorkCounter
from pySDC.implementations.datatype_classes.mesh import mesh, imex_mesh


# ---------------------------------------------------------------------------
# Module-level helper
# ---------------------------------------------------------------------------

def _reaction(u, eps, dw):
    r"""
    Allen-Cahn reaction: :math:`-(2/\varepsilon^2)\,u(1-u)(1-2u) - 6\,d_w\,u(1-u)`.

    Parameters
    ----------
    u : numpy.ndarray
        Solution values.
    eps : float
        Interface-width parameter.
    dw : float
        Driving-force parameter.

    Returns
    -------
    numpy.ndarray
    """
    return -2.0 / eps**2 * u * (1.0 - u) * (1.0 - 2.0 * u) - 6.0 * dw * u * (1.0 - u)


# ---------------------------------------------------------------------------
# Shared base class
# ---------------------------------------------------------------------------

class _AllenCahn1D_Base(Problem):
    r"""
    Shared setup for all problem classes.

    Builds a fourth-order FD Laplacian on :math:`[0,1]` with *zero* Dirichlet
    BCs (the boundary correction, if any, is handled by the subclass).

    **Spatial discretisation**

    Interior rows :math:`k = 2, \ldots, n-3` use the standard centred
    fourth-order stencil

    .. math::

        u_{xx}(x_k) \approx
        \frac{-u_{k-2} + 16u_{k-1} - 30u_k + 16u_{k+1} - u_{k+2}}{12\,\Delta x^2}.

    The two innermost rows (:math:`k = 1` and :math:`k = n-2`) use the same
    centred stencil with the known Dirichlet boundary value substituted for
    the out-of-domain point, which contributes to the :math:`b_\text{bc}`
    correction in the inhomogeneous subclass.

    The two outermost rows (:math:`k = 0` and :math:`k = n-1`) use a
    6-point one-sided stencil derived to maintain fourth-order accuracy
    without requiring a ghost point outside the domain:

    .. math::

        u_{xx}(x_0) \approx
        \frac{10u_L - 15u_0 - 4u_1 + 14u_2 - 6u_3 + u_4}{12\,\Delta x^2},

    and symmetrically at :math:`k = n-1`.  The :math:`u_L` and :math:`u_R`
    terms contribute to :math:`b_\text{bc}` in the inhomogeneous subclass;
    for the homogeneous and lifted cases they vanish.

    Parameters
    ----------
    nvars : int
        Number of interior grid points (default 127; must be ≥ 5).
    eps : float
        Interface-width parameter :math:`\varepsilon` (default 1.0).
    dw : float
        Driving-force parameter (default 0.0).

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

    def __init__(self, nvars=127, eps=1.0, dw=0.0):
        if nvars < 5:
            raise ValueError(f"nvars must be >= 5 for the 4th-order FD Laplacian; got {nvars}")
        super().__init__((nvars, None, np.dtype('float64')))
        self._makeAttributeAndRegister('nvars', 'eps', 'dw', localVars=locals(), readOnly=True)

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

        # Row 0: 6-point one-sided stencil (4th-order, no ghost point needed).
        # u_xx(x_0) ~ (10*u_L - 15*u[0] - 4*u[1] + 14*u[2] - 6*u[3] + u[4]) / (12*dx^2)
        # Matrix part (u_L = 0 for zero-BC; non-zero contribution goes to b_bc):
        A[0, 0] = -15
        A[0, 1] = -4
        A[0, 2] = 14
        A[0, 3] = -6
        A[0, 4] = 1

        # Row 1: standard 4th-order centred, uses u[-1] = u_L (boundary) as b_bc correction.
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

        # Row n-2: standard 4th-order centred, uses u[n] = u_R (boundary) as b_bc correction.
        # u_xx(x_{n-2}) ~ (-u[n-4] + 16*u[n-3] - 30*u[n-2] + 16*u[n-1] - u_R) / (12*dx^2)
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


# ---------------------------------------------------------------------------
# Case 1: Homogeneous BCs
# ---------------------------------------------------------------------------

class allencahn_1d_hom(_AllenCahn1D_Base):
    r"""
    Allen-Cahn problem with **homogeneous** Dirichlet BCs.

    **Exact solution**

    .. math::

        u_\text{ex}(x, t) = \sin(\pi x)\,\cos(t), \quad x \in [0, 1].

    **Boundary conditions**

    :math:`u(0, t) = 0`,  :math:`u(1, t) = 0`  (time-independent, homogeneous).

    **Forcing term**

    .. math::

        g(x, t)
        = \partial_t u_\text{ex} - \partial_{xx} u_\text{ex} - R(u_\text{ex})
        = -\sin(\pi x)\sin(t) + \pi^2 \sin(\pi x)\cos(t) - R(u_\text{ex}),

    so that the modified Allen-Cahn PDE is satisfied exactly.

    **IMEX split**

    * :math:`f_\text{impl} = A\,u`
      – autonomous Laplacian, no boundary-correction vector needed.
    * :math:`f_\text{expl} = R(u) + g(x, t)`

    This is the *cleanest* case: no time-dependent BCs, no :math:`b_\text{bc}`.
    It tests only the effect of the nonlinear reaction on the SDC convergence
    order.
    """

    def _forcing(self, t):
        r"""
        Forcing term :math:`g(x_i, t)` at interior grid points.

        .. math::

            g = -\sin(\pi x)\sin(t) + \pi^2\sin(\pi x)\cos(t) - R(u_\text{ex}).
        """
        x = self.xvalues
        u_ex = np.sin(np.pi * x) * np.cos(t)
        u_t = -np.sin(np.pi * x) * np.sin(t)
        u_xx = -np.pi**2 * np.sin(np.pi * x) * np.cos(t)
        return u_t - u_xx - _reaction(u_ex, self.eps, self.dw)

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
            ``f.impl = A u``, ``f.expl = R(u) + g(t)``.
        """
        f = self.dtype_f(self.init)
        f.impl[:] = self.A.dot(u)
        f.expl[:] = _reaction(np.asarray(u), self.eps, self.dw) + self._forcing(t)
        self.work_counters['rhs']()
        return f

    def solve_system(self, rhs, factor, u0, t):
        r"""
        Solve :math:`(I - \text{factor}\cdot A)\,u = \text{rhs}`.

        No boundary-correction term (homogeneous BCs).

        Parameters
        ----------
        rhs : dtype_u
            Right-hand side from the sweeper.
        factor : float
            Implicit step size.
        u0 : dtype_u
            Initial guess (unused; direct solver).
        t : float
            Current time (unused).

        Returns
        -------
        me : dtype_u
        """
        me = self.dtype_u(self.init)
        me[:] = spsolve(sp.eye(self.nvars, format='csc') - factor * self.A, np.asarray(rhs))
        return me

    def u_exact(self, t):
        r"""
        Exact solution :math:`\sin(\pi x)\cos(t)` at interior grid points.

        Parameters
        ----------
        t : float

        Returns
        -------
        me : dtype_u
        """
        me = self.dtype_u(self.init, val=0.0)
        me[:] = np.sin(np.pi * self.xvalues) * np.cos(t)
        return me


# ---------------------------------------------------------------------------
# Case 2: Inhomogeneous BCs, standard b_bc correction
# ---------------------------------------------------------------------------

class allencahn_1d_inhom(_AllenCahn1D_Base):
    r"""
    Allen-Cahn problem with **time-dependent** Dirichlet BCs,
    handled by the standard boundary-correction vector :math:`b_\text{bc}(t)`.

    **Exact solution**

    .. math::

        u_\text{ex}(x, t) = \cos(\pi x)\,\cos(t), \quad x \in [0, 1].

    **Boundary conditions**

    :math:`u(0, t) = \cos(t)`,  :math:`u(1, t) = -\cos(t)`.

    **Forcing term**

    .. math::

        g(x, t)
        = -\cos(\pi x)\sin(t) + \pi^2\cos(\pi x)\cos(t) - R(u_\text{ex}).

    **IMEX split**

    * :math:`f_\text{impl} = A\,u + b_\text{bc}(t)`
      – includes the time-dependent boundary-correction vector.
    * :math:`f_\text{expl} = R(u) + g(x, t)`

    The :math:`b_\text{bc}(t)` correction makes :math:`f_\text{impl}`
    explicitly time-dependent, which may cause order reduction compared
    to the homogeneous case.
    """

    def _bc_vector(self, t):
        r"""
        Boundary-correction vector :math:`b_\text{bc}(t)` for the 4th-order
        FD Laplacian.

        The non-zero entries come from substituting the Dirichlet boundary
        values into the FD stencil rows that touch the domain boundary:

        * Row 0 (6-point one-sided stencil):
          :math:`b_0 = 10\cos(t) / (12\,\Delta x^2)`.
        * Row 1 (standard centred stencil, needs :math:`u_L`):
          :math:`b_1 = -\cos(t) / (12\,\Delta x^2)`.
        * Row :math:`n-2` (standard centred, needs :math:`u_R`):
          :math:`b_{n-2} = \cos(t) / (12\,\Delta x^2)`.
        * Row :math:`n-1` (6-point one-sided):
          :math:`b_{n-1} = -10\cos(t) / (12\,\Delta x^2)`.

        Parameters
        ----------
        t : float

        Returns
        -------
        numpy.ndarray
        """
        n = self.nvars
        inv12dx2 = 1.0 / (12.0 * self.dx**2)
        u_L = np.cos(t)
        u_R = -np.cos(t)
        bc = np.zeros(n)
        bc[0] = 10 * u_L * inv12dx2
        bc[1] = -u_L * inv12dx2
        bc[n - 2] = -u_R * inv12dx2
        bc[n - 1] = 10 * u_R * inv12dx2
        return bc

    def _forcing(self, t):
        r"""
        Forcing :math:`g = -\cos(\pi x)\sin(t) + \pi^2\cos(\pi x)\cos(t) - R(u_\text{ex})`.

        Parameters
        ----------
        t : float

        Returns
        -------
        numpy.ndarray
        """
        x = self.xvalues
        u_ex = np.cos(np.pi * x) * np.cos(t)
        u_t = -np.cos(np.pi * x) * np.sin(t)
        u_xx = -np.pi**2 * np.cos(np.pi * x) * np.cos(t)
        return u_t - u_xx - _reaction(u_ex, self.eps, self.dw)

    def eval_f(self, u, t):
        r"""
        Evaluate :math:`f_\text{impl} = A u + b_\text{bc}(t)` and
        :math:`f_\text{expl} = R(u) + g(t)`.

        Parameters
        ----------
        u : dtype_u
        t : float

        Returns
        -------
        f : imex_mesh
        """
        f = self.dtype_f(self.init)
        f.impl[:] = self.A.dot(u) + self._bc_vector(t)
        f.expl[:] = _reaction(np.asarray(u), self.eps, self.dw) + self._forcing(t)
        self.work_counters['rhs']()
        return f

    def solve_system(self, rhs, factor, u0, t):
        r"""
        Solve :math:`(I - \text{factor}\cdot A)\,u = \text{rhs} + \text{factor}\cdot b_\text{bc}(t)`.

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
        me[:] = spsolve(sp.eye(self.nvars, format='csc') - factor * self.A, system_rhs)
        return me

    def u_exact(self, t):
        r"""
        Exact solution :math:`\cos(\pi x)\cos(t)` at interior grid points.

        Parameters
        ----------
        t : float

        Returns
        -------
        me : dtype_u
        """
        me = self.dtype_u(self.init, val=0.0)
        me[:] = np.cos(np.pi * self.xvalues) * np.cos(t)
        return me


# ---------------------------------------------------------------------------
# Case 3: Inhomogeneous BCs, boundary lifting
# ---------------------------------------------------------------------------

class allencahn_1d_inhom_lift(_AllenCahn1D_Base):
    r"""
    Allen-Cahn problem with **time-dependent** Dirichlet BCs treated
    by **boundary lifting**.

    **Background**

    The same exact solution as :class:`allencahn_1d_inhom` is used, but
    reformulated in terms of a lifted variable :math:`v = u - E(t)` where

    .. math::

        E(x, t) = (1 - 2x)\cos(t)

    is a linear interpolant satisfying the BCs:
    :math:`E(0,t) = \cos(t)` and :math:`E(1,t) = -\cos(t)`.

    The lifted variable satisfies *homogeneous* BCs and evolves according to

    .. math::

        v_t = A\,v + R(v+E) + g(x,t) - \dot{E}(t),

    where :math:`\dot{E}(x,t) = -(1-2x)\sin(t)`.  The implicit operator
    :math:`A` is now **purely autonomous** (no :math:`b_\text{bc}` correction),
    which should restore the full SDC convergence order if the BC treatment
    was the cause of the stalling.

    **Exact lifted solution**

    .. math::

        v_\text{ex}(x, t) = u_\text{ex}(x,t) - E(x,t)
        = \cos(t)\bigl(\cos(\pi x) - 1 + 2x\bigr).

    Parameters
    ----------
    nvars, eps, dw : see :class:`_AllenCahn1D_Base`.
    """

    def lift(self, t):
        r"""
        Lift :math:`E(x_i, t) = (1 - 2 x_i)\cos(t)`.

        Parameters
        ----------
        t : float

        Returns
        -------
        numpy.ndarray
        """
        return (1.0 - 2.0 * self.xvalues) * np.cos(t)

    def _dlift_dt(self, t):
        r"""
        :math:`\dot{E}(x_i, t) = -(1-2 x_i)\sin(t)`.

        Parameters
        ----------
        t : float

        Returns
        -------
        numpy.ndarray
        """
        return -(1.0 - 2.0 * self.xvalues) * np.sin(t)

    def _forcing(self, t):
        r"""
        Physical forcing :math:`g(x,t) = u_t - u_{xx} - R(u_\text{ex})`.

        Parameters
        ----------
        t : float

        Returns
        -------
        numpy.ndarray
        """
        x = self.xvalues
        u_ex = np.cos(np.pi * x) * np.cos(t)
        u_t = -np.cos(np.pi * x) * np.sin(t)
        u_xx = -np.pi**2 * np.cos(np.pi * x) * np.cos(t)
        return u_t - u_xx - _reaction(u_ex, self.eps, self.dw)

    def eval_f(self, v, t):
        r"""
        Evaluate the RHS for the lifted variable :math:`v`.

        Parameters
        ----------
        v : dtype_u
            Current lifted solution.
        t : float

        Returns
        -------
        f : imex_mesh
            ``f.impl = A v`` (no BC correction),
            ``f.expl = R(v+E) + g(t) - dE/dt``.
        """
        E = self.lift(t)
        u = np.asarray(v) + E  # recover physical variable

        f = self.dtype_f(self.init)
        f.impl[:] = self.A.dot(v)
        f.expl[:] = _reaction(u, self.eps, self.dw) + self._forcing(t) - self._dlift_dt(t)
        self.work_counters['rhs']()
        return f

    def solve_system(self, rhs, factor, u0, t):
        r"""
        Solve :math:`(I - \text{factor}\cdot A)\,v = \text{rhs}`.

        No boundary-correction term needed since :math:`v` satisfies
        homogeneous BCs.

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
        me[:] = spsolve(sp.eye(self.nvars, format='csc') - factor * self.A, np.asarray(rhs))
        return me

    def u_exact(self, t):
        r"""
        Exact lifted solution :math:`v_\text{ex} = u_\text{ex} - E(t)`.

        To recover the physical solution, call :meth:`lift` and add.

        Parameters
        ----------
        t : float

        Returns
        -------
        me : dtype_u
        """
        me = self.dtype_u(self.init, val=0.0)
        me[:] = np.cos(np.pi * self.xvalues) * np.cos(t) - self.lift(t)
        return me
