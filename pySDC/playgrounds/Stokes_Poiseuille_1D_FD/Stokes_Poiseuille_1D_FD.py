r"""
1-D unsteady Stokes / Poiseuille problem – semi-explicit index-1 DAE
=====================================================================

Problem
-------
The 1-D unsteady Stokes equations on :math:`y \in [0, 1]` with a global
flow-rate constraint read

.. math::

    u_t = \nu\,u_{yy} + G(t) + f(y, t), \qquad u(0,t)=u(1,t)=0,

.. math::

    \int_0^1 u\,\mathrm{d}y = q(t),

where :math:`G(t)` is the unknown pressure gradient and :math:`q(t)` is a
prescribed flow-rate.  After finite-difference discretisation this becomes
the semi-explicit index-1 DAE

.. math::

    \mathbf{u}' = \nu A\,\mathbf{u} + G\,\mathbf{1} + \mathbf{f}(t),

.. math::

    0 = B\,\mathbf{u} - q(t),

where :math:`A` is the 4th-order FD Laplacian,
:math:`B = h\,\mathbf{1}^T` (rectangle-rule integral, :math:`h = 1/(N+1)`),
:math:`\mathbf{1}` is the vector of ones, and :math:`G(t)` is the Lagrange
multiplier for the flow-rate constraint.

Manufactured solution
---------------------
.. math::

    u_\text{ex}(y, t) = \sin(\pi y)\,\sin(t), \quad G_\text{ex}(t) = \cos(t).

Forcing:

.. math::

    f(y, t) = \sin(\pi y)\cos(t) + \nu\pi^2\sin(\pi y)\sin(t) - \cos(t).

State and sweeper
-----------------
The state variable uses :class:`~pySDC.projects.DAE.misc.meshDAE.MeshDAE`
with ``nvars`` interior points:

* ``u.diff[:]`` – velocity on :math:`N` interior grid points.
* ``u.alg[0]`` – pressure gradient :math:`G` (Lagrange multiplier).

The compatible sweeper is
:class:`~pySDC.projects.DAE.sweepers.semiImplicitDAE.SemiImplicitDAE`.

Without constraint lifting (class :class:`stokes_poiseuille_1d_fd`), the
constraint :math:`B\mathbf{u} = q(t)` has a time-dependent right-hand side,
which causes order reduction in :math:`G` to order :math:`M` (= 3 for
3 RADAU-RIGHT nodes).

Constraint lifting (class :class:`stokes_poiseuille_1d_fd_lift`)
-----------------------------------------------------------------
Introduce the lifting function

.. math::

    \mathbf{u}_\ell(t) = \frac{q(t)}{s}\,\mathbf{1}, \qquad
    s = B\mathbf{1} = h N,

which satisfies :math:`B\mathbf{u}_\ell(t) = q(t)` exactly.  Let
:math:`\tilde{\mathbf{v}} = \mathbf{u} - \mathbf{u}_\ell(t)`.  The lifted
variable satisfies the **homogeneous** (autonomous) constraint

.. math::

    0 = B\,\tilde{\mathbf{v}},

and evolves according to

.. math::

    \tilde{\mathbf{v}}' = \nu A\,\tilde{\mathbf{v}}
                        + \bigl[\nu A\,\mathbf{u}_\ell(t)
                               + \mathbf{f}(t)
                               - \dot{\mathbf{u}}_\ell(t)\bigr]
                        + G\,\mathbf{1}.

Because the constraint :math:`B\tilde{\mathbf{v}} = 0` no longer contains a
time-dependent right-hand side, the Lagrange multiplier :math:`G` is expected
to converge at the full collocation order :math:`2M-1`, matching the velocity.

Classes
-------
stokes_poiseuille_1d_fd
    No lifting; constraint :math:`B\mathbf{u} = q(t)` (time-dependent).
    Pressure converges at order :math:`M`.

stokes_poiseuille_1d_fd_lift
    Constraint lifting; homogeneous :math:`B\tilde{\mathbf{v}} = 0`.
    Expected to restore full order :math:`2M-1` in the pressure.
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

from pySDC.projects.DAE.misc.problemDAE import ProblemDAE


# ---------------------------------------------------------------------------
# Fourth-order FD Laplacian (same stencil as AllenCahn_1D_FD)
# ---------------------------------------------------------------------------

def _build_laplacian(n, dx):
    r"""
    Assemble the fourth-order FD Laplacian on *n* interior points with
    **zero** Dirichlet boundary conditions.

    Interior rows use the centred stencil
    :math:`(-u_{k-2}+16u_{k-1}-30u_k+16u_{k+1}-u_{k+2})/(12\Delta x^2)`;
    the outermost rows use a 6-point one-sided stencil.  With homogeneous
    BCs the boundary-correction vector :math:`b_\text{bc}` is identically
    zero.

    Parameters
    ----------
    n : int
        Number of interior grid points (must be ≥ 5).
    dx : float
        Grid spacing.

    Returns
    -------
    scipy.sparse.csc_matrix
    """
    inv12dx2 = 1.0 / (12.0 * dx**2)
    L = sp.lil_matrix((n, n))

    # Row 0: 6-point one-sided stencil (u_L = 0).
    L[0, 0] = -15
    L[0, 1] = -4
    L[0, 2] = 14
    L[0, 3] = -6
    L[0, 4] = 1

    # Row 1: standard centred stencil; u_{-1} = 0.
    L[1, 0] = 16
    L[1, 1] = -30
    L[1, 2] = 16
    L[1, 3] = -1

    # Interior rows 2 … n-3.
    for k in range(2, n - 2):
        L[k, k - 2] = -1
        L[k, k - 1] = 16
        L[k, k] = -30
        L[k, k + 1] = 16
        L[k, k + 2] = -1

    # Row n-2: standard centred stencil; u_n = 0.
    L[n - 2, n - 4] = -1
    L[n - 2, n - 3] = 16
    L[n - 2, n - 2] = -30
    L[n - 2, n - 1] = 16

    # Row n-1: mirror of row 0 (u_R = 0).
    L[n - 1, n - 5] = 1
    L[n - 1, n - 4] = -6
    L[n - 1, n - 3] = 14
    L[n - 1, n - 2] = -4
    L[n - 1, n - 1] = -15

    return (L * inv12dx2).tocsc()


# ---------------------------------------------------------------------------
# Shared base class
# ---------------------------------------------------------------------------

class _StokesBase(ProblemDAE):
    r"""
    Shared setup for the Stokes/Poiseuille problem classes.

    Builds the 4th-order FD Laplacian, grid coordinates, and the
    manufactured-forcing helpers.

    Parameters
    ----------
    nvars : int
        Number of interior grid points (must be ≥ 5).
    nu : float
        Kinematic viscosity.
    newton_tol : float
        Tolerance passed to :class:`~pySDC.projects.DAE.misc.problemDAE.ProblemDAE`.

    Attributes
    ----------
    dx : float
        Grid spacing :math:`1/(N+1)`.
    xvalues : numpy.ndarray
        Interior grid-point :math:`y`-coordinates.
    A : scipy.sparse.csc_matrix
        :math:`\nu`-scaled fourth-order FD Laplacian.
    ones : numpy.ndarray
        Vector of ones, shape ``(nvars,)``.
    C_B : float
        :math:`h\sum_i \sin(\pi y_i)` — coefficient in
        :math:`q(t) = C_B \sin(t)`.
    s : float
        :math:`B \mathbf{1} = h N` — discrete integral of the all-ones vector.
    """

    def __init__(self, nvars, nu, newton_tol):
        if nvars < 5:
            raise ValueError(
                f'nvars must be >= 5 for the 4th-order FD Laplacian; got {nvars}'
            )
        super().__init__(nvars=nvars, newton_tol=newton_tol)
        self._makeAttributeAndRegister('nvars', 'nu', localVars=locals(), readOnly=True)

        self.dx = 1.0 / (nvars + 1)
        self.xvalues = np.linspace(self.dx, 1.0 - self.dx, nvars)

        # nu-scaled Laplacian (4th-order FD, zero BCs)
        self.A = nu * _build_laplacian(nvars, self.dx)

        # Discrete-integral operator B = h * 1^T (used as a plain vector)
        self.ones = np.ones(nvars)
        self.C_B = self.dx * float(np.sum(np.sin(np.pi * self.xvalues)))
        self.s = self.dx * nvars  # B * ones = h * N

    def _q(self, t):
        r"""
        Flow-rate constraint RHS: :math:`q(t) = C_B\,\sin(t)`.
        """
        return self.C_B * np.sin(t)

    def _forcing(self, t):
        r"""
        Manufactured forcing consistent with :math:`u_\text{ex}` and
        :math:`G_\text{ex} = \cos(t)`:

        .. math::

            f(y, t) = \sin(\pi y)\cos(t)
                    + \nu\pi^2\sin(\pi y)\sin(t) - \cos(t).
        """
        y = self.xvalues
        return (
            np.sin(np.pi * y) * np.cos(t)
            + self.nu * np.pi**2 * np.sin(np.pi * y) * np.sin(t)
            - np.cos(t) * self.ones
        )

    def _B_dot(self, v):
        r"""Rectangle-rule integral: :math:`B \mathbf{v} = h\sum_i v_i`."""
        return self.dx * float(np.sum(v))

    def _schur_solve(self, rhs_eff, v_approx, factor, constraint_rhs):
        r"""
        Schur-complement saddle-point solve.

        Finds :math:`(\mathbf{U}, G)` satisfying

        .. math::

            (I - \alpha\nu A)\,\mathbf{U} - G\,\mathbf{1} = \mathbf{r}_\text{eff},

        .. math::

            B(\mathbf{v}_\text{approx} + \alpha\,\mathbf{U}) = c,

        where :math:`c` is ``constraint_rhs`` (``q(t)`` for the standard
        formulation, ``0`` for the lifted formulation).

        Parameters
        ----------
        rhs_eff : numpy.ndarray
            Effective velocity RHS :math:`\mathbf{r}_\text{eff}`.
        v_approx : numpy.ndarray
            Current velocity approximation at the node.
        factor : float
            Implicit prefactor :math:`\alpha = \Delta t\,\tilde{q}_{mm}`.
        constraint_rhs : float
            Right-hand side of the constraint equation.

        Returns
        -------
        U : numpy.ndarray
            Velocity derivative at the node.
        G_new : float
            Pressure gradient (Lagrange multiplier).
        """
        M = sp.eye(self.nvars, format='csc') - factor * self.A
        w = spsolve(M, rhs_eff)
        v0 = spsolve(M, self.ones)

        Bw = self._B_dot(w)
        # Bv0 is positive because (I - factor*A) is an M-matrix for typical
        # factor values, so its inverse has positive row-sums, giving B*v0 > 0.
        Bv0 = self._B_dot(v0)
        Bv = self._B_dot(v_approx)
        G_new = (constraint_rhs - Bv - factor * Bw) / (factor * Bv0)

        U = w + G_new * v0
        return U, float(G_new)


# ---------------------------------------------------------------------------
# Case 1: No lifting – constraint B·u = q(t)
# ---------------------------------------------------------------------------

class stokes_poiseuille_1d_fd(_StokesBase):
    r"""
    1-D Stokes/Poiseuille DAE **without** constraint lifting.

    The constraint :math:`B\mathbf{u} = q(t)` has a time-dependent RHS,
    which causes **order reduction** in the pressure gradient :math:`G` to
    order :math:`M` (number of collocation nodes) instead of the full
    collocation order :math:`2M-1`.

    **Exact solution** (manufactured):

    .. math::

        u_\text{ex}(y_i, t) = \sin(\pi y_i)\,\sin(t), \quad
        G_\text{ex}(t) = \cos(t).

    Parameters
    ----------
    nvars : int
        Number of interior grid points (default 127; must be ≥ 5).
    nu : float
        Kinematic viscosity (default 1.0).
    newton_tol : float
        Unused; passed to base class (default 1e-10).
    """

    def __init__(self, nvars=127, nu=1.0, newton_tol=1e-10):
        super().__init__(nvars=nvars, nu=nu, newton_tol=newton_tol)

    def eval_f(self, u, du, t):
        r"""
        Fully-implicit DAE residual:

        .. math::

            F_\text{diff} = \mathbf{u}' - \nu A\,\mathbf{u}
                          - G\,\mathbf{1} - \mathbf{f}(t),

        .. math::

            F_\text{alg} = B\,\mathbf{u} - q(t).
        """
        f = self.dtype_f(self.init, val=0.0)
        u_vel = np.asarray(u.diff)
        du_vel = np.asarray(du.diff)
        G = float(u.alg[0])

        f.diff[:] = du_vel - (self.A.dot(u_vel) + G * self.ones + self._forcing(t))
        f.alg[0] = self._B_dot(u_vel) - self._q(t)

        self.work_counters['rhs']()
        return f

    def solve_system(self, impl_sys, u_approx, factor, u0, t):
        r"""
        Schur-complement solve with constraint :math:`B\mathbf{u} = q(t)`.

        Parameters
        ----------
        impl_sys : callable
            Unused; system solved directly.
        u_approx : MeshDAE
            Current velocity approximation at the node.
        factor : float
            Implicit prefactor :math:`\alpha`.
        u0 : MeshDAE
            Unused (direct solver).
        t : float
            Current time.

        Returns
        -------
        me : MeshDAE
            ``me.diff[:]`` = velocity derivative :math:`\mathbf{U}_m`,
            ``me.alg[0]`` = pressure gradient :math:`G_m`.
        """
        me = self.dtype_u(self.init, val=0.0)
        v_approx = np.asarray(u_approx.diff).copy()

        rhs_eff = self.A.dot(v_approx) + self._forcing(t)
        U, G_new = self._schur_solve(rhs_eff, v_approx, factor, self._q(t))

        me.diff[:] = U
        me.alg[0] = G_new
        return me

    def u_exact(self, t):
        r"""
        Exact solution: ``diff`` = :math:`\sin(\pi y)\sin(t)`,
        ``alg[0]`` = :math:`\cos(t)`.
        """
        me = self.dtype_u(self.init, val=0.0)
        me.diff[:] = np.sin(np.pi * self.xvalues) * np.sin(t)
        me.alg[0] = np.cos(t)
        return me

    def du_exact(self, t):
        r"""
        Exact time derivative: ``diff`` = :math:`\sin(\pi y)\cos(t)`,
        ``alg[0]`` = :math:`-\sin(t)`.
        """
        me = self.dtype_u(self.init, val=0.0)
        me.diff[:] = np.sin(np.pi * self.xvalues) * np.cos(t)
        me.alg[0] = -np.sin(t)
        return me


# ---------------------------------------------------------------------------
# Case 2: Constraint lifting – homogeneous constraint B·ṽ = 0
# ---------------------------------------------------------------------------

class stokes_poiseuille_1d_fd_lift(_StokesBase):
    r"""
    1-D Stokes/Poiseuille DAE **with** constraint lifting.

    **Lifting function**

    .. math::

        \mathbf{u}_\ell(t) = \frac{q(t)}{s}\,\mathbf{1}, \qquad
        s = B\mathbf{1} = h N,

    satisfies :math:`B\mathbf{u}_\ell = q(t)` exactly.  The lifted
    variable :math:`\tilde{\mathbf{v}} = \mathbf{u} - \mathbf{u}_\ell(t)`
    satisfies the **homogeneous** constraint

    .. math::

        0 = B\,\tilde{\mathbf{v}},

    and evolves according to

    .. math::

        \tilde{\mathbf{v}}' = \nu A\,\tilde{\mathbf{v}} + G\,\mathbf{1}
                            + \bigl[\nu A\,\mathbf{u}_\ell(t)
                                   + \mathbf{f}(t)
                                   - \dot{\mathbf{u}}_\ell(t)\bigr].

    Because :math:`B\tilde{\mathbf{v}} = 0` is autonomous (no time
    dependence), the Lagrange multiplier :math:`G` is expected to converge
    at the full collocation order :math:`2M-1`, matching the velocity.

    **State variable** ``u``:

    * ``u.diff[:]`` = :math:`\tilde{\mathbf{v}}` (lifted velocity).
    * ``u.alg[0]`` = :math:`G`.

    **Exact lifted solution**:

    .. math::

        \tilde{\mathbf{v}}_\text{ex}(y_i, t) =
            \sin(\pi y_i)\sin(t) - \mathbf{u}_\ell(t),
        \quad G_\text{ex}(t) = \cos(t).

    Parameters
    ----------
    nvars : int
        Number of interior grid points (default 127; must be ≥ 5).
    nu : float
        Kinematic viscosity (default 1.0).
    newton_tol : float
        Unused; passed to base class (default 1e-10).
    """

    def __init__(self, nvars=127, nu=1.0, newton_tol=1e-10):
        super().__init__(nvars=nvars, nu=nu, newton_tol=newton_tol)
        # Precompute A*ones (needed in eval_f and solve_system).
        self._A_ones = np.asarray(self.A.dot(self.ones)).copy()

    # ------------------------------------------------------------------
    # Lifting helpers
    # ------------------------------------------------------------------

    def lift(self, t):
        r"""
        Lifting function :math:`\mathbf{u}_\ell(t) = (q(t)/s)\,\mathbf{1}`.

        Satisfies :math:`B\mathbf{u}_\ell(t) = q(t)` exactly.

        Parameters
        ----------
        t : float

        Returns
        -------
        numpy.ndarray, shape (nvars,)
        """
        return (self.C_B * np.sin(t) / self.s) * self.ones

    def _lift_dot(self, t):
        r"""Time derivative :math:`\dot{\mathbf{u}}_\ell(t) = (C_B\cos(t)/s)\,\mathbf{1}`."""
        return (self.C_B * np.cos(t) / self.s) * self.ones

    def _A_lift(self, t):
        r"""
        :math:`\nu A\,\mathbf{u}_\ell(t) = (C_B\sin(t)/s)\,A\mathbf{1}`.

        Non-zero only at boundary rows (boundary-correction effect of the
        spatially-constant lift profile).
        """
        return (self.C_B * np.sin(t) / self.s) * self._A_ones

    def _net_forcing(self, t):
        r"""
        Net explicit forcing for the lifted system:

        .. math::

            \mathbf{F}_\text{net}(t) = \nu A\,\mathbf{u}_\ell(t)
                                      + \mathbf{f}(t)
                                      - \dot{\mathbf{u}}_\ell(t).
        """
        return self._A_lift(t) + self._forcing(t) - self._lift_dot(t)

    # ------------------------------------------------------------------
    # pySDC / SemiImplicitDAE interface
    # ------------------------------------------------------------------

    def eval_f(self, v, dv, t):
        r"""
        Fully-implicit DAE residual for the lifted variable
        :math:`\tilde{\mathbf{v}}`:

        .. math::

            F_\text{diff} = \tilde{\mathbf{v}}' - \nu A\,\tilde{\mathbf{v}}
                          - G\,\mathbf{1} - \mathbf{F}_\text{net}(t),

        .. math::

            F_\text{alg} = B\,\tilde{\mathbf{v}} \quad
            (\text{homogeneous, zero at exact solution}).
        """
        f = self.dtype_f(self.init, val=0.0)
        v_vel = np.asarray(v.diff)
        dv_vel = np.asarray(dv.diff)
        G = float(v.alg[0])

        f.diff[:] = dv_vel - (
            self.A.dot(v_vel) + G * self.ones + self._net_forcing(t)
        )
        f.alg[0] = self._B_dot(v_vel)  # B·ṽ (should vanish at exact solution)

        self.work_counters['rhs']()
        return f

    def solve_system(self, impl_sys, v_approx, factor, u0, t):
        r"""
        Schur-complement solve with **homogeneous** constraint
        :math:`B(\tilde{\mathbf{v}}_\text{approx} + \alpha\,\tilde{\mathbf{U}}) = 0`.

        Parameters
        ----------
        impl_sys : callable
            Unused; system solved directly.
        v_approx : MeshDAE
            Current lifted velocity approximation at the node.
        factor : float
            Implicit prefactor :math:`\alpha`.
        u0 : MeshDAE
            Unused (direct solver).
        t : float
            Current time.

        Returns
        -------
        me : MeshDAE
            ``me.diff[:]`` = lifted velocity derivative :math:`\tilde{\mathbf{U}}_m`,
            ``me.alg[0]`` = pressure gradient :math:`G_m`.
        """
        me = self.dtype_u(self.init, val=0.0)
        vv = np.asarray(v_approx.diff).copy()

        rhs_eff = self.A.dot(vv) + self._net_forcing(t)
        U, G_new = self._schur_solve(rhs_eff, vv, factor, 0.0)

        me.diff[:] = U
        me.alg[0] = G_new
        return me

    def u_exact(self, t):
        r"""
        Exact lifted solution at time :math:`t`:

        ``diff`` = :math:`\sin(\pi y)\sin(t) - \mathbf{u}_\ell(t)`,
        ``alg[0]`` = :math:`\cos(t)`.
        """
        me = self.dtype_u(self.init, val=0.0)
        me.diff[:] = np.sin(np.pi * self.xvalues) * np.sin(t) - self.lift(t)
        me.alg[0] = np.cos(t)
        return me

    def du_exact(self, t):
        r"""
        Exact time derivative of the lifted solution:

        ``diff`` = :math:`\sin(\pi y)\cos(t) - \dot{\mathbf{u}}_\ell(t)`,
        ``alg[0]`` = :math:`-\sin(t)`.
        """
        me = self.dtype_u(self.init, val=0.0)
        me.diff[:] = np.sin(np.pi * self.xvalues) * np.cos(t) - self._lift_dot(t)
        me.alg[0] = -np.sin(t)
        return me
