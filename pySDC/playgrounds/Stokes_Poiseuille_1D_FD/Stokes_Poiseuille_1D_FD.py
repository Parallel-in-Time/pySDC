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

Two sweepers are supported:

* :class:`~pySDC.projects.DAE.sweepers.semiImplicitDAE.SemiImplicitDAE`
  (U-formulation): velocity order :math:`M+1`, pressure order :math:`M`
  (standard), approaching :math:`M+1` (lifted), or order :math:`M+2`
  (differentiated constraint — coincides with :math:`2M-1` only for
  :math:`M = 3`).

Three constraint treatments are provided — see Classes section below.

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

Because the constraint :math:`B\tilde{\mathbf{v}} = 0` is autonomous, the
order reduction in :math:`G` is reduced (pressure approaches :math:`M+1`)
but not fully eliminated in the SDC context.

Differentiated constraint (class :class:`stokes_poiseuille_1d_fd_diffconstr`)
------------------------------------------------------------------------------
Instead of enforcing the algebraic constraint
:math:`B\mathbf{u}_m = q(\tau_m)` at each SDC stage, enforce its time
derivative :math:`B\mathbf{U}_m = q'(\tau_m)`.  The Schur formula becomes

.. math::

    G_m = \frac{q'(\tau_m) - B\mathbf{w}}{B\mathbf{v}_0},

and the stage pressure error reduces from :math:`\mathcal{O}(\Delta t^M)`
to :math:`\mathcal{O}(\Delta t^{M+1})`.  The U-formulation quadrature then
gives endpoint error :math:`\Delta t \cdot \mathcal{O}(\Delta t^{M+1}) =
\mathcal{O}(\Delta t^{M+2})`.  **For :math:`M = 3`, :math:`M+2 = 5 = 2M-1`
coincidentally equals the full RADAU collocation order.**  For :math:`M = 4`,
the order is :math:`M+2 = 6 \neq 2M-1 = 7`, as confirmed numerically.
Achieving :math:`2M-1` for :math:`M \geq 4` requires the y-formulation
(standard RADAU-IIA), not the U-formulation used here.

Classes
-------
stokes_poiseuille_1d_fd
    No lifting; algebraic constraint :math:`B\mathbf{u} = q(t)`.
    SemiImplicitDAE: vel :math:`M+1`, pres :math:`M`.

stokes_poiseuille_1d_fd_lift
    Constraint lifting; homogeneous :math:`B\tilde{\mathbf{v}} = 0`.
    SemiImplicitDAE: vel :math:`M+1`, pres approaching :math:`M+1`.

stokes_poiseuille_1d_fd_diffconstr
    Differentiated constraint :math:`B\mathbf{U}_m = q'(\tau_m)`.
    SemiImplicitDAE: vel and pres both at :math:`M+2` (= :math:`2M-1`
    only for :math:`M = 3`).

stokes_poiseuille_1d_fd_lift_diffconstr
    Lifting + differentiated :math:`B\tilde{\mathbf{U}} = 0`.
    Equivalent to :class:`stokes_poiseuille_1d_fd_lift` at the fixed point.

stokes_poiseuille_1d_fd_full
    No lifting, FullyImplicitDAE (same fixed point as standard).

stokes_poiseuille_1d_fd_lift_full
    Lifting, FullyImplicitDAE (same fixed point as lifted).

stokes_poiseuille_1d_fd_coupled
    Explicit :math:`(N+1)\times(N+1)` block solve for the differentiated
    constraint (mathematically equivalent to diffconstr Schur).
    Optional ``project=True`` also enforces :math:`B\mathbf{u}_m = q(\tau_m)`
    via a post-solve projection, but this **degrades** convergence because
    it creates an inconsistency between ``solve_system`` (which after
    projection enforces the algebraic constraint) and ``eval_f`` (which
    checks the differentiated constraint).  Provided for pedagogical
    comparison only.
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

    def _q_prime(self, t):
        r"""
        Time derivative of the flow-rate RHS: :math:`q'(t) = C_B\,\cos(t)`.
        """
        return self.C_B * np.cos(t)

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

    def _schur_solve_full_implicit(self, rhs_eff, v_approx, factor, constraint_rhs):
        r"""
        Schur-complement saddle-point solve for use with
        :class:`~pySDC.projects.DAE.sweepers.fullyImplicitDAE.FullyImplicitDAE`.

        Finds :math:`(\mathbf{U}, G')` satisfying

        .. math::

            (I - \alpha\nu A)\,\mathbf{U} - \alpha G'\,\mathbf{1}
                = \mathbf{r}_\text{eff},

        .. math::

            B(\mathbf{v}_\text{approx} + \alpha\,\mathbf{U}) = c,

        where :math:`G' = \mathrm{d}G/\mathrm{d}t` is the **derivative** of
        the pressure gradient and :math:`c` is ``constraint_rhs``.

        Compared to :meth:`_schur_solve`, here ``rhs_eff`` must already
        include the :math:`G_0\,\mathbf{1}` term (the current pressure
        estimate from ``u_approx.alg[0]``), and the denominator carries an
        extra :math:`\alpha` factor:

        .. math::

            G' = \frac{c - B\mathbf{v} - \alpha B\mathbf{w}}
                      {\alpha^2 B\mathbf{v}_0},

        Parameters
        ----------
        rhs_eff : numpy.ndarray
            Effective velocity RHS :math:`\nu A\mathbf{v} + G_0\mathbf{1}
            + \mathbf{f}_\text{net}`.
        v_approx : numpy.ndarray
            Current velocity approximation at the node.
        factor : float
            Implicit prefactor :math:`\alpha = \Delta t\,\tilde{q}_{mm}`.
        constraint_rhs : float
            RHS of the constraint :math:`c` (``q(t)`` standard; ``0`` lifted).

        Returns
        -------
        U : numpy.ndarray
            Velocity derivative at the node.
        G_prime : float
            Derivative of the pressure gradient :math:`G'`.
        """
        M = sp.eye(self.nvars, format='csc') - factor * self.A
        w = spsolve(M, rhs_eff)
        v0 = spsolve(M, self.ones)

        Bw = self._B_dot(w)
        Bv0 = self._B_dot(v0)
        Bv = self._B_dot(v_approx)
        # Extra factor of alpha in denominator vs _schur_solve
        denom = factor**2 * Bv0
        assert abs(denom) > 0.0, (
            f'_schur_solve_full_implicit: denominator factor²·B·v₀ = {denom:.3e} is zero; '
            f'factor = {factor}, B·v₀ = {Bv0:.3e}'
        )
        G_prime = (constraint_rhs - Bv - factor * Bw) / denom

        U = w + factor * G_prime * v0
        return U, float(G_prime)

    def _coupled_block_solve(self, rhs_eff, v_approx, factor, q_prime_val,
                             project=False, q_val=None):
        r"""
        Fully-coupled :math:`(N+1)\times(N+1)` block solve for the
        differentiated constraint :math:`B\mathbf{U} = q'(t)`.

        Assembles and solves the full block system

        .. math::

            \begin{pmatrix}
                I - \alpha\nu A & -\mathbf{1} \\
                h\,\mathbf{1}^T & 0
            \end{pmatrix}
            \begin{pmatrix} \mathbf{U} \\ G \end{pmatrix}
            =
            \begin{pmatrix} \mathbf{r}_\text{eff} \\ q'(t) \end{pmatrix}

        This is **mathematically equivalent** to
        :meth:`_schur_solve_diffconstr` (which reduces the same system to a
        scalar Schur equation), but solves the full :math:`(N+1)` sparse
        system directly.

        If ``project=True``, a post-solve projection step enforces the
        **original** algebraic constraint :math:`B(\mathbf{v}+\alpha\mathbf{U})
        = q(t)` in addition to the differentiated one.  The minimal-norm
        correction

        .. math::

            \delta = -\frac{B(\mathbf{v}+\alpha\mathbf{U}) - q(t)}{s}
                       \,\mathbf{1}

        is added to :math:`\mathbf{v}+\alpha\mathbf{U}`, giving

        .. math::

            \mathbf{U}_\text{proj} =
                \mathbf{U} + \frac{\delta}{\alpha}.

        The pressure :math:`G` is then updated by solving the Schur formula
        :math:`G = (q'(t) - B\mathbf{U}_\text{proj}) / (B\mathbf{v}_0)` so
        that :math:`B\mathbf{U}_\text{proj}` uses the corrected derivative.

        Parameters
        ----------
        rhs_eff : numpy.ndarray
            Effective velocity RHS
            :math:`\nu A\mathbf{v}_\text{approx} + \mathbf{f}(t)`.
        v_approx : numpy.ndarray
            Current velocity approximation at the node.
        factor : float
            Implicit prefactor :math:`\alpha = \Delta t\,\tilde{q}_{mm}`.
        q_prime_val : float
            Value of :math:`q'(t)` at the current stage time.
        project : bool, optional
            If ``True``, apply the projection step that also enforces
            :math:`B(\mathbf{v}+\alpha\mathbf{U}) = q(t)`.  Default ``False``.
        q_val : float or None
            Value of :math:`q(t)` required when ``project=True``.

        Returns
        -------
        U : numpy.ndarray
            Velocity derivative at the node.
        G_new : float
            Pressure gradient satisfying :math:`B\mathbf{U} = q'(t)` (before
            projection) or consistent with the projected velocity (after).
        """
        n = self.nvars
        top_left = sp.eye(n, format='csc') - factor * self.A
        top_right = sp.csc_matrix(-self.ones.reshape(-1, 1))
        bot_left = sp.csc_matrix(self.dx * self.ones.reshape(1, -1))
        bot_right = sp.csc_matrix(np.zeros((1, 1)))

        K = sp.bmat(
            [[top_left, top_right], [bot_left, bot_right]],
            format='csc',
        )
        rhs = np.concatenate([rhs_eff, [q_prime_val]])
        sol = spsolve(K, rhs)

        U = sol[:n].copy()
        G = float(sol[n])

        if project and q_val is not None:
            # Post-solve projection: enforce B*(v + factor*U) = q(t).
            # WARNING: this projection changes U_m in a way that is
            # INCONSISTENT with eval_f, which checks B*u' - q'(t) = 0.
            # After projection, B*U_proj = B*U - violation/factor ≠ q'(t)
            # in general (since B*U = q'(t) before projection but violation ≠ 0).
            # As a result, the SDC residual no longer converges cleanly and
            # the sweep converges to a DIFFERENT (worse) fixed point.
            # The projection is provided here for pedagogical comparison only;
            # it should NOT be used in practice with the differentiated-constraint
            # eval_f.
            u_m = v_approx + factor * U
            violation = self._B_dot(u_m) - q_val
            if abs(violation) > 0.0:
                delta = -(violation / self.s) * self.ones
                U = U + delta / factor
                # Update G from the momentum residual.
                # At the unperturbed solution, (I-factor*A)*U - G*ones = rhs_eff,
                # so G*ones = (I-factor*A)*U - rhs_eff.  After projection, U changes
                # and all N components of (I-factor*A)*U_proj - rhs_eff theoretically
                # equal the same G value; we take the mean for numerical stability.
                G = float(np.mean(top_left.dot(U) - rhs_eff))

        return U, G

    def _schur_solve_diffconstr(self, rhs_eff, factor, q_prime_val):
        r"""
        Schur-complement solve using the **differentiated constraint**
        :math:`B\mathbf{U} = q'(t)`.

        Instead of enforcing the original algebraic constraint
        :math:`B(\mathbf{v} + \alpha\mathbf{U}) = q(t)`, this method uses
        the differentiated form :math:`B\mathbf{U} = q'(t)`.  The key
        formula is

        .. math::

            G = \frac{q'(t) - B\mathbf{w}}{B\mathbf{v}_0},

        where
        :math:`\mathbf{w} = (I - \alpha\nu A)^{-1}\mathbf{r}_\text{eff}` and
        :math:`\mathbf{v}_0 = (I - \alpha\nu A)^{-1}\mathbf{1}`.

        **Why this gives higher-order pressure**: At the SDC fixed point the
        stage velocities satisfy :math:`\mathbf{u}_m - \mathbf{u}(\tau_m) =
        \mathcal{O}(\Delta t^{M+1})` (collocation accuracy).  The
        differentiated-constraint error propagates as

        .. math::

            e_{G_m} = G_m - G(\tau_m) = -\frac{B A\,e_{\mathbf{u}_m}}{s}
                    = \mathcal{O}(\Delta t^{M+1}),

        whereas the original algebraic constraint gives only
        :math:`\mathcal{O}(\Delta t^M)`.

        Parameters
        ----------
        rhs_eff : numpy.ndarray
            Effective velocity RHS
            :math:`\nu A\mathbf{v}_\text{approx} + \mathbf{f}_\text{net}(t)`.
        factor : float
            Implicit prefactor :math:`\alpha = \Delta t\,\tilde{q}_{mm}`.
        q_prime_val : float
            Value of :math:`q'(t)` at the current stage time.

        Returns
        -------
        U : numpy.ndarray
            Velocity derivative at the node.
        G_new : float
            Pressure gradient satisfying :math:`B\mathbf{U} = q'(t)`.
        """
        M_mat = sp.eye(self.nvars, format='csc') - factor * self.A
        w = spsolve(M_mat, rhs_eff)
        v0 = spsolve(M_mat, self.ones)

        Bw = self._B_dot(w)
        Bv0 = self._B_dot(v0)
        assert abs(Bv0) > 0.0, (
            f'_schur_solve_diffconstr: B·v₀ = {Bv0:.3e} is zero; factor = {factor}'
        )
        G_new = (q_prime_val - Bw) / Bv0

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


# ---------------------------------------------------------------------------
# Case 3: FullyImplicitDAE – no lifting
# ---------------------------------------------------------------------------

class stokes_poiseuille_1d_fd_full(stokes_poiseuille_1d_fd):
    r"""
    1-D Stokes/Poiseuille DAE **without** constraint lifting, compatible with
    :class:`~pySDC.projects.DAE.sweepers.fullyImplicitDAE.FullyImplicitDAE`.

    The key difference from :class:`stokes_poiseuille_1d_fd` is in
    ``solve_system``: here the unknown is the full derivative
    :math:`(\mathbf{U}, G') = (\mathbf{u}', G')`, i.e. ``me.alg[0]`` is the
    *derivative* of the pressure gradient.  The pressure gradient at the
    node is then recovered by quadrature:

    .. math::

        G_m = G_0 + \Delta t \sum_{j=1}^{M} Q_{mj}\,G'_j,

    which gives the pressure the same collocation structure as
    the velocity (both recovered via the quadrature formula).

    .. note::

        In practice, :class:`FullyImplicitDAE` and
        :class:`~.semiImplicitDAE.SemiImplicitDAE` converge to the
        **same collocation fixed point** for this index-1 DAE: the
        :math:`\mathcal{O}(\Delta t^M)` errors in the stage pressure
        values break the :math:`2M-1` superconvergence, and both sweepers
        achieve velocity order :math:`M+1` and pressure order :math:`M`
        (standard) or increasing toward :math:`M+1` (lifted).  The
        :class:`FullyImplicitDAE` formulation is included for completeness
        and pedagogical comparison.

    The Schur-complement solve for the unknown :math:`(\mathbf{U}, G')`:

    .. math::

        (I - \alpha\nu A)\,\mathbf{U} - \alpha G'\,\mathbf{1}
            = \nu A\mathbf{v} + G_0\mathbf{1} + \mathbf{f}(t),

    .. math::

        B(\mathbf{v} + \alpha\,\mathbf{U}) = q(t),

    yields

    .. math::

        G' = \frac{q(t) - B\mathbf{v} - \alpha B\mathbf{w}}{\alpha^2 B\mathbf{v}_0},
        \quad
        \mathbf{U} = \mathbf{w} + \alpha G'\,\mathbf{v}_0,

    where :math:`\mathbf{w} = (I-\alpha\nu A)^{-1}(\nu A\mathbf{v}
    + G_0\mathbf{1} + \mathbf{f})` and
    :math:`\mathbf{v}_0 = (I-\alpha\nu A)^{-1}\mathbf{1}`.

    Parameters
    ----------
    nvars : int
        Number of interior grid points (default 127; must be ≥ 5).
    nu : float
        Kinematic viscosity (default 1.0).
    newton_tol : float
        Unused; passed to base class (default 1e-10).
    """

    def solve_system(self, impl_sys, u_approx, factor, u0, t):
        r"""
        Schur-complement solve for
        :class:`~pySDC.projects.DAE.sweepers.fullyImplicitDAE.FullyImplicitDAE`.

        Returns the **derivative** :math:`(\mathbf{U}, G')`.
        ``me.alg[0]`` = :math:`G'` (pressure-gradient time derivative).

        Parameters
        ----------
        impl_sys : callable
            Unused; system solved directly.
        u_approx : MeshDAE
            Approximation :math:`(\mathbf{v}, G_0)` at the current node.
        factor : float
            Implicit prefactor :math:`\alpha`.
        u0 : MeshDAE
            Unused (direct solver).
        t : float
            Current time.

        Returns
        -------
        me : MeshDAE
            ``me.diff[:]`` = :math:`\mathbf{U}`,
            ``me.alg[0]``  = :math:`G'`.
        """
        me = self.dtype_u(self.init, val=0.0)
        v_approx = np.asarray(u_approx.diff).copy()
        G0 = float(u_approx.alg[0])

        # rhs_eff includes current G0: FullyImplicitDAE treats G as a differential
        # variable (G = G0 + factor*G'), so G0 enters the RHS (unlike SemiImplicitDAE
        # where u_approx.alg = 0 and G is purely algebraic in the local solve).
        rhs_eff = self.A.dot(v_approx) + G0 * self.ones + self._forcing(t)
        U, G_prime = self._schur_solve_full_implicit(rhs_eff, v_approx, factor, self._q(t))

        me.diff[:] = U
        me.alg[0] = G_prime
        return me


# ---------------------------------------------------------------------------
# Case 4: FullyImplicitDAE – with lifting
# ---------------------------------------------------------------------------

class stokes_poiseuille_1d_fd_lift_full(stokes_poiseuille_1d_fd_lift):
    r"""
    1-D Stokes/Poiseuille DAE **with** constraint lifting, compatible with
    :class:`~pySDC.projects.DAE.sweepers.fullyImplicitDAE.FullyImplicitDAE`.

    Combines the homogeneous constraint :math:`B\tilde{\mathbf{v}} = 0`
    from :class:`stokes_poiseuille_1d_fd_lift` with the
    :class:`FullyImplicitDAE`-consistent ``solve_system`` from
    :class:`stokes_poiseuille_1d_fd_full`.

    .. note::

        :class:`FullyImplicitDAE` and :class:`SemiImplicitDAE` converge
        to the same collocation fixed point for this DAE: velocity order
        :math:`M+1` and pressure order increasing toward :math:`M+1`
        (same as :class:`stokes_poiseuille_1d_fd_lift`).  This class is
        provided for pedagogical comparison.

    Parameters
    ----------
    nvars : int
        Number of interior grid points (default 127; must be ≥ 5).
    nu : float
        Kinematic viscosity (default 1.0).
    newton_tol : float
        Unused; passed to base class (default 1e-10).
    """

    def solve_system(self, impl_sys, v_approx_mesh, factor, u0, t):
        r"""
        Schur-complement solve for
        :class:`~pySDC.projects.DAE.sweepers.fullyImplicitDAE.FullyImplicitDAE`
        with the **homogeneous** constraint :math:`B\tilde{\mathbf{v}} = 0`.

        Returns :math:`(\tilde{\mathbf{U}}, G')`.

        Parameters
        ----------
        impl_sys : callable
            Unused; system solved directly.
        v_approx_mesh : MeshDAE
            Approximation :math:`(\tilde{\mathbf{v}}, G_0)` at the node.
        factor : float
            Implicit prefactor :math:`\alpha`.
        u0 : MeshDAE
            Unused (direct solver).
        t : float
            Current time.

        Returns
        -------
        me : MeshDAE
            ``me.diff[:]`` = :math:`\tilde{\mathbf{U}}`,
            ``me.alg[0]``  = :math:`G'`.
        """
        me = self.dtype_u(self.init, val=0.0)
        vv = np.asarray(v_approx_mesh.diff).copy()
        G0 = float(v_approx_mesh.alg[0])

        rhs_eff = self.A.dot(vv) + G0 * self.ones + self._net_forcing(t)
        U, G_prime = self._schur_solve_full_implicit(rhs_eff, vv, factor, 0.0)

        me.diff[:] = U
        me.alg[0] = G_prime
        return me


# ---------------------------------------------------------------------------
# Case 5: No lifting – differentiated constraint B·U = q'(t)
# ---------------------------------------------------------------------------

class stokes_poiseuille_1d_fd_diffconstr(stokes_poiseuille_1d_fd):
    r"""
    1-D Stokes/Poiseuille DAE using the **differentiated constraint**
    :math:`B\mathbf{u}' = q'(t)` at every SDC stage.

    This is a remedy for the order reduction in the pressure gradient.
    Rather than enforcing the original algebraic constraint
    :math:`B\mathbf{u}_m = q(\tau_m)` (which limits :math:`G` to order
    :math:`M`), each stage solve uses

    .. math::

        B\,\mathbf{U}_m = q'(\tau_m),

    where :math:`\mathbf{U}_m = \mathbf{u}'(\tau_m)` is the velocity
    derivative.  The corresponding Schur-complement formula is

    .. math::

        G_m = \frac{q'(\tau_m) - B\mathbf{w}}{B\mathbf{v}_0},

    giving pressure error
    :math:`e_{G_m} = -BA\,e_{\mathbf{u}_m}/s = \mathcal{O}(\Delta t^{M+1})`
    (one order higher than the algebraic constraint).

    ``eval_f`` is also modified to check :math:`B\mathbf{u}' - q'(t) = 0`
    (so that the SDC residual converges to machine precision at the fixed
    point of the differentiated-constraint iteration).

    .. note::

        The original constraint :math:`B\mathbf{u} = q(t)` is satisfied
        approximately: since :math:`B\mathbf{u}(t_n) = q(t_n)` (consistent
        IC) and :math:`B\mathbf{U}_m = q'(\tau_m)` at every stage,
        the quadrature formula gives
        :math:`B\mathbf{u}_{n+1} - q(t_{n+1}) = \mathcal{O}(\Delta t^{2M})`
        at the endpoint (Gauss quadrature error) and
        :math:`\mathcal{O}(\Delta t^{M+1})` at interior nodes.

    Parameters
    ----------
    nvars : int
        Number of interior grid points (default 127; must be ≥ 5).
    nu : float
        Kinematic viscosity (default 1.0).
    newton_tol : float
        Unused; passed to base class (default 1e-10).
    """

    def eval_f(self, u, du, t):
        r"""
        Fully-implicit DAE residual using the **differentiated constraint**:

        .. math::

            F_\text{diff} = \mathbf{u}' - \nu A\,\mathbf{u}
                          - G\,\mathbf{1} - \mathbf{f}(t),

        .. math::

            F_\text{alg} = B\,\mathbf{u}' - q'(t).

        The second equation uses the velocity **derivative** ``du.diff``
        instead of the velocity state ``u.diff``, making the SDC residual
        exactly zero (machine precision) when :meth:`solve_system` has
        enforced :math:`B\mathbf{U}_m = q'(\tau_m)`.
        """
        f = self.dtype_f(self.init, val=0.0)
        u_vel = np.asarray(u.diff)
        du_vel = np.asarray(du.diff)
        G = float(u.alg[0])

        f.diff[:] = du_vel - (self.A.dot(u_vel) + G * self.ones + self._forcing(t))
        f.alg[0] = self._B_dot(du_vel) - self._q_prime(t)

        self.work_counters['rhs']()
        return f

    def solve_system(self, impl_sys, u_approx, factor, u0, t):
        r"""
        Schur-complement solve using the differentiated constraint
        :math:`B\mathbf{U} = q'(t)`.

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
            ``me.diff[:]`` = velocity derivative :math:`\mathbf{U}_m`
            satisfying :math:`B\mathbf{U}_m = q'(t)`,
            ``me.alg[0]`` = pressure gradient :math:`G_m`.
        """
        me = self.dtype_u(self.init, val=0.0)
        v_approx = np.asarray(u_approx.diff).copy()

        rhs_eff = self.A.dot(v_approx) + self._forcing(t)
        U, G_new = self._schur_solve_diffconstr(rhs_eff, factor, self._q_prime(t))

        me.diff[:] = U
        me.alg[0] = G_new
        return me


# ---------------------------------------------------------------------------
# Case 6: Lifting + differentiated constraint B·Ũ = 0
# ---------------------------------------------------------------------------

class stokes_poiseuille_1d_fd_lift_diffconstr(stokes_poiseuille_1d_fd_lift):
    r"""
    1-D Stokes/Poiseuille DAE with **constraint lifting** and the
    **differentiated constraint** :math:`B\tilde{\mathbf{u}}' = 0`.

    Combines:

    * The homogeneous constraint :math:`B\tilde{\mathbf{v}} = 0` from
      :class:`stokes_poiseuille_1d_fd_lift` (reduces order reduction).
    * The differentiated constraint :math:`B\tilde{\mathbf{U}} = 0` in
      the stage solve, analogous to :class:`stokes_poiseuille_1d_fd_diffconstr`.

    .. note::

        For the lifted problem the original constraint is
        :math:`B\tilde{\mathbf{v}} = 0` (homogeneous, constant in time).
        Its time derivative is :math:`B\tilde{\mathbf{v}}' = 0`, which is
        the same condition.  The differentiated-constraint Schur solve
        therefore reduces to
        :math:`G = -B\mathbf{w}_\text{net} / (B\mathbf{v}_0)`,
        which is **equivalent to the original lifted Schur solve at the fixed
        point** (when :math:`B\tilde{\mathbf{v}} \approx 0`).  Both this class
        and :class:`stokes_poiseuille_1d_fd_lift` therefore converge to the
        same fixed point and give identical convergence orders.  This class is
        included for completeness and to confirm the equivalence.

    Parameters
    ----------
    nvars : int
        Number of interior grid points (default 127; must be ≥ 5).
    nu : float
        Kinematic viscosity (default 1.0).
    newton_tol : float
        Unused; passed to base class (default 1e-10).
    """

    def eval_f(self, u, du, t):
        r"""
        Fully-implicit DAE residual using the differentiated homogeneous
        constraint :math:`B\tilde{\mathbf{u}}' = 0`:

        .. math::

            F_\text{diff} = \tilde{\mathbf{u}}' - \nu A\,\tilde{\mathbf{v}}
                          - G\,\mathbf{1} - \mathbf{f}_\text{net}(t),

        .. math::

            F_\text{alg} = B\,\tilde{\mathbf{u}}' - 0.
        """
        f = self.dtype_f(self.init, val=0.0)
        v_tilde = np.asarray(u.diff)
        dv_tilde = np.asarray(du.diff)
        G = float(u.alg[0])

        f.diff[:] = dv_tilde - (self.A.dot(v_tilde) + G * self.ones + self._net_forcing(t))
        f.alg[0] = self._B_dot(dv_tilde)   # B·ũ' = 0

        self.work_counters['rhs']()
        return f

    def solve_system(self, impl_sys, u_approx, factor, u0, t):
        r"""
        Schur-complement solve using the differentiated homogeneous
        constraint :math:`B\tilde{\mathbf{U}} = 0`.

        Parameters
        ----------
        impl_sys : callable
            Unused; system solved directly.
        u_approx : MeshDAE
            Current velocity approximation :math:`(\tilde{\mathbf{v}}, G)`.
        factor : float
            Implicit prefactor :math:`\alpha`.
        u0 : MeshDAE
            Unused (direct solver).
        t : float
            Current time.

        Returns
        -------
        me : MeshDAE
            ``me.diff[:]`` = lifted velocity derivative
            :math:`\tilde{\mathbf{U}}_m` (satisfies
            :math:`B\tilde{\mathbf{U}}_m = 0`),
            ``me.alg[0]`` = pressure gradient :math:`G_m`.
        """
        me = self.dtype_u(self.init, val=0.0)
        v_approx = np.asarray(u_approx.diff).copy()

        rhs_eff = self.A.dot(v_approx) + self._net_forcing(t)
        # Differentiated homogeneous constraint: B·U_tilde = 0 → q'_eff = 0
        U, G_new = self._schur_solve_diffconstr(rhs_eff, factor, 0.0)

        me.diff[:] = U
        me.alg[0] = G_new
        return me


# ---------------------------------------------------------------------------
# Case 7: Coupled block solve + projection
# ---------------------------------------------------------------------------

class stokes_poiseuille_1d_fd_coupled(stokes_poiseuille_1d_fd):
    r"""
    1-D Stokes/Poiseuille DAE using an **explicit** :math:`(N+1)\times(N+1)`
    block solve for the differentiated constraint, with an optional
    post-solve projection that also enforces the original algebraic constraint.

    Two sub-variants are provided via the ``project`` constructor parameter:

    **Block solve only** (``project=False``, default)
        Assembles and solves the full :math:`(N+1)\times(N+1)` sparse system

        .. math::

            \begin{pmatrix}
                I - \alpha\nu A & -\mathbf{1} \\
                h\,\mathbf{1}^T & 0
            \end{pmatrix}
            \begin{pmatrix} \mathbf{U} \\ G \end{pmatrix}
            =
            \begin{pmatrix}
                \nu A\mathbf{v} + \mathbf{f}(\tau_m) \\
                q'(\tau_m)
            \end{pmatrix}

        This is **mathematically equivalent** to
        :class:`stokes_poiseuille_1d_fd_diffconstr` (which reduces the same
        system to a scalar Schur equation).  Convergence orders are the same:
        velocity :math:`M+2`, pressure :math:`M+2` (:math:`= 2M-1` for
        :math:`M = 3` by coincidence).

    **Block solve + projection** (``project=True``)
        After the block solve, a minimal-norm correction enforces the
        **original** algebraic constraint :math:`B(\mathbf{v}+\alpha\mathbf{U})
        = q(\tau_m)` as well.

        .. warning::

            This variant gives **worse** results than the plain block solve.
            The root cause is an inconsistency between ``solve_system`` (which
            after projection enforces :math:`B(\mathbf{v}+\alpha\mathbf{U})
            = q(t)`) and ``eval_f`` (which checks the **differentiated**
            constraint :math:`B\mathbf{u}' - q'(t) = 0`).  Because the
            projection changes :math:`\mathbf{U}` so that
            :math:`B\mathbf{U} \neq q'(t)` any more, the SDC residual
            never converges cleanly and the sweep converges to a different,
            lower-accuracy fixed point.  Numerically, the projection variant
            achieves only velocity :math:`M+1 \approx 4`, pressure
            :math:`M \approx 3` — the same as the standard algebraic formulation.

        The lesson is that **self-consistency between** ``solve_system`` **and**
        ``eval_f`` **is essential**: both must enforce the same constraint
        (either the algebraic :math:`B\mathbf{u}=q` or the differentiated
        :math:`B\mathbf{u}'=q'`).  Mixing the two degrades convergence.

    The ``eval_f`` uses the differentiated constraint
    :math:`F_\text{alg} = B\mathbf{u}' - q'(t) = 0`, matching
    :class:`stokes_poiseuille_1d_fd_diffconstr`.

    Parameters
    ----------
    nvars : int
        Number of interior grid points (default 127; must be ≥ 5).
    nu : float
        Kinematic viscosity (default 1.0).
    newton_tol : float
        Unused; passed to base class (default 1e-10).
    project : bool
        If ``True``, apply the post-solve projection step that also enforces
        :math:`B(\mathbf{v}+\alpha\mathbf{U}) = q(\tau_m)`.  Default ``False``.
        See warning above — this is provided for pedagogical comparison only.
    """

    def __init__(self, nvars=127, nu=1.0, newton_tol=1e-10, project=False):
        super().__init__(nvars=nvars, nu=nu, newton_tol=newton_tol)
        self._makeAttributeAndRegister('project', localVars=locals(), readOnly=True)

    def eval_f(self, u, du, t):
        r"""
        Fully-implicit DAE residual using the **differentiated constraint**:

        .. math::

            F_\text{diff} = \mathbf{u}' - \nu A\,\mathbf{u}
                          - G\,\mathbf{1} - \mathbf{f}(t),

        .. math::

            F_\text{alg} = B\,\mathbf{u}' - q'(t).

        Identical to :class:`stokes_poiseuille_1d_fd_diffconstr`.
        """
        f = self.dtype_f(self.init, val=0.0)
        u_vel = np.asarray(u.diff)
        du_vel = np.asarray(du.diff)
        G = float(u.alg[0])

        f.diff[:] = du_vel - (self.A.dot(u_vel) + G * self.ones + self._forcing(t))
        f.alg[0] = self._B_dot(du_vel) - self._q_prime(t)

        self.work_counters['rhs']()
        return f

    def solve_system(self, impl_sys, u_approx, factor, u0, t):
        r"""
        Coupled :math:`(N+1)\times(N+1)` block solve with the differentiated
        constraint :math:`B\mathbf{U} = q'(t)`.

        Optionally applies a post-solve projection onto
        :math:`B(\mathbf{v}+\alpha\mathbf{U}) = q(t)` if ``self.project``
        is ``True``.

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
        U, G_new = self._coupled_block_solve(
            rhs_eff, v_approx, factor, self._q_prime(t),
            project=self.project,
            q_val=self._q(t) if self.project else None,
        )

        me.diff[:] = U
        me.alg[0] = G_new
        return me
