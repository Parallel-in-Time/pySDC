r"""
Manufactured-solution (MMS) problem classes for the 1D Allen-Cahn playground
==============================================================================

Three self-contained problem classes on the domain :math:`[0, 1]`, each
implementing the Allen-Cahn equation with a source term that enforces a
prescribed smooth exact solution:

.. math::

    u_t = u_{xx} + R(u) + g(x, t),

where

.. math::

    R(u) = -\frac{2}{\varepsilon^2}\,u(1-u)(1-2u) - 6\,d_w\,u(1-u)

and :math:`g(x,t)` is chosen so that :math:`u_\text{mms}` is the exact
solution.  The three classes differ only in their boundary treatment:

* :class:`allencahn_1d_mms_hom` – homogeneous Dirichlet BCs; no
  boundary-correction vector :math:`b_\text{bc}` required.
  The implicit operator :math:`f_\text{impl} = A u` is autonomous.
* :class:`allencahn_1d_mms_inhom` – time-dependent Dirichlet BCs handled
  via the standard :math:`b_\text{bc}(t)` correction in
  :math:`f_\text{impl}`.
* :class:`allencahn_1d_mms_inhom_lift` – same time-dependent BCs treated
  by boundary lifting (:math:`v = u - E`); the implicit operator
  :math:`f_\text{impl} = A v` is again autonomous.

Comparing the three cases under fully-converged IMEX-SDC shows that
including a time-dependent :math:`b_\text{bc}(t)` in :math:`f_\text{impl}`
reduces the collocation order, while boundary lifting restores it.

Classes
-------
allencahn_1d_mms_hom
    :math:`u_\text{mms}(x,t) = \sin(\pi x)\cos(t)`,
    homogeneous BCs :math:`u|_\partial = 0`.

allencahn_1d_mms_inhom
    :math:`u_\text{mms}(x,t) = \cos(\pi x)\cos(t)`,
    time-dependent BCs :math:`u(0,t)=\cos(t)`,
    :math:`u(1,t)=-\cos(t)`.  Standard :math:`b_\text{bc}(t)` correction.

allencahn_1d_mms_inhom_lift
    Same exact solution as :class:`allencahn_1d_mms_inhom` but reformulated
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

class _AllenCahn1D_MMS_Base(Problem):
    r"""
    Shared setup for all MMS problem classes.

    Builds a uniform FD Laplacian on :math:`[0,1]` with *zero* Dirichlet
    BCs (the boundary correction, if any, is handled by the subclass).

    Parameters
    ----------
    nvars : int
        Number of interior grid points (default 127).
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
        Tridiagonal FD Laplacian (with *zero* boundary rows, not the BC
        correction).
    """

    dtype_u = mesh
    dtype_f = imex_mesh

    def __init__(self, nvars=127, eps=1.0, dw=0.0):
        super().__init__((nvars, None, np.dtype('float64')))
        self._makeAttributeAndRegister('nvars', 'eps', 'dw', localVars=locals(), readOnly=True)

        self.dx = 1.0 / (nvars + 1)
        self.xvalues = np.linspace(self.dx, 1.0 - self.dx, nvars)

        diag_main = np.full(nvars, -2.0)
        diag_off = np.ones(nvars - 1)
        self.A = (
            sp.diags([diag_off, diag_main, diag_off], offsets=[-1, 0, 1],
                     shape=(nvars, nvars), format='csc') / self.dx**2
        )
        self.work_counters['rhs'] = WorkCounter()


# ---------------------------------------------------------------------------
# Case 1: Homogeneous BCs
# ---------------------------------------------------------------------------

class allencahn_1d_mms_hom(_AllenCahn1D_MMS_Base):
    r"""
    Allen-Cahn MMS problem with **homogeneous** Dirichlet BCs.

    **Exact solution**

    .. math::

        u_\text{mms}(x, t) = \sin(\pi x)\,\cos(t), \quad x \in [0, 1].

    **Boundary conditions**

    :math:`u(0, t) = 0`,  :math:`u(1, t) = 0`  (time-independent, homogeneous).

    **Forcing term**

    .. math::

        g(x, t)
        = \partial_t u_\text{mms} - \partial_{xx} u_\text{mms} - R(u_\text{mms})
        = -\sin(\pi x)\sin(t) + \pi^2 \sin(\pi x)\cos(t) - R(u_\text{mms}),

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

            g = -\sin(\pi x)\sin(t) + \pi^2\sin(\pi x)\cos(t) - R(u_\text{mms}).
        """
        x = self.xvalues
        u_mms = np.sin(np.pi * x) * np.cos(t)
        u_t = -np.sin(np.pi * x) * np.sin(t)
        u_xx = -np.pi**2 * np.sin(np.pi * x) * np.cos(t)
        return u_t - u_xx - _reaction(u_mms, self.eps, self.dw)

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

class allencahn_1d_mms_inhom(_AllenCahn1D_MMS_Base):
    r"""
    Allen-Cahn MMS problem with **time-dependent** Dirichlet BCs,
    handled by the standard boundary-correction vector :math:`b_\text{bc}(t)`.

    **Exact solution**

    .. math::

        u_\text{mms}(x, t) = \cos(\pi x)\,\cos(t), \quad x \in [0, 1].

    **Boundary conditions**

    :math:`u(0, t) = \cos(t)`,  :math:`u(1, t) = -\cos(t)`.

    **Forcing term**

    .. math::

        g(x, t)
        = -\cos(\pi x)\sin(t) + \pi^2\cos(\pi x)\cos(t) - R(u_\text{mms}).

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
        Boundary-correction vector :math:`b_\text{bc}(t)`.

        .. math::

            b_0 = \cos(t)/\Delta x^2, \quad
            b_{n-1} = -\cos(t)/\Delta x^2, \quad
            b_i = 0 \text{ otherwise}.

        Parameters
        ----------
        t : float

        Returns
        -------
        numpy.ndarray
        """
        bc = np.zeros(self.nvars)
        bc[0] = np.cos(t) / self.dx**2
        bc[-1] = -np.cos(t) / self.dx**2
        return bc

    def _forcing(self, t):
        r"""
        Forcing :math:`g = -\cos(\pi x)\sin(t) + \pi^2\cos(\pi x)\cos(t) - R(u_\text{mms})`.

        Parameters
        ----------
        t : float

        Returns
        -------
        numpy.ndarray
        """
        x = self.xvalues
        u_mms = np.cos(np.pi * x) * np.cos(t)
        u_t = -np.cos(np.pi * x) * np.sin(t)
        u_xx = -np.pi**2 * np.cos(np.pi * x) * np.cos(t)
        return u_t - u_xx - _reaction(u_mms, self.eps, self.dw)

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

class allencahn_1d_mms_inhom_lift(_AllenCahn1D_MMS_Base):
    r"""
    Allen-Cahn MMS problem with **time-dependent** Dirichlet BCs treated
    by **boundary lifting**.

    **Background**

    The same exact solution as :class:`allencahn_1d_mms_inhom` is used, but
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

        v_\text{mms}(x, t) = u_\text{mms}(x,t) - E(x,t)
        = \cos(t)\bigl(\cos(\pi x) - 1 + 2x\bigr).

    Parameters
    ----------
    nvars, eps, dw : see :class:`_AllenCahn1D_MMS_Base`.
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
        Physical forcing :math:`g(x,t) = u_t - u_{xx} - R(u_\text{mms})`.

        Parameters
        ----------
        t : float

        Returns
        -------
        numpy.ndarray
        """
        x = self.xvalues
        u_mms = np.cos(np.pi * x) * np.cos(t)
        u_t = -np.cos(np.pi * x) * np.sin(t)
        u_xx = -np.pi**2 * np.cos(np.pi * x) * np.cos(t)
        return u_t - u_xx - _reaction(u_mms, self.eps, self.dw)

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
        Exact lifted solution :math:`v_\text{mms} = u_\text{mms} - E(t)`.

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
