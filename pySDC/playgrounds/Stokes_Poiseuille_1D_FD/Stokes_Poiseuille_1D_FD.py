r"""
1-D unsteady Stokes / Poiseuille problem – semi-explicit index-1 DAE
=====================================================================

Problem
-------
The 1-D unsteady Stokes equations on :math:`y \in [0, 1]` with a global
incompressibility constraint read

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

    u_\text{ex}(y, t) = \sin(\pi y)\,\sin(t), \quad
    G_\text{ex}(t) = \cos(t).

The manufactured forcing consistent with this exact solution is

.. math::

    f(y, t) = \sin(\pi y)\cos(t) + \nu\pi^2\sin(\pi y)\sin(t) - \cos(t).

State and sweeper
-----------------
The state variable uses :class:`~pySDC.projects.DAE.misc.meshDAE.MeshDAE`
initialised with ``nvars`` interior points:

* ``u.diff[:]`` – velocity on :math:`N` interior grid points.
* ``u.alg[0]`` – pressure gradient :math:`G`; ``u.alg[1:]`` is unused.

The compatible sweeper is
:class:`~pySDC.projects.DAE.sweepers.semiImplicitDAE.SemiImplicitDAE`,
which keeps the *U-formulation*: ``L.f[m].diff`` stores the velocity
derivative :math:`U_m = u'(t_m)`, and ``L.u[m].diff`` is reconstructed by
time-integration.  Only the differential part is integrated; the algebraic
variable :math:`G` is enforced at every SDC node via ``solve_system``.

Saddle-point solve
------------------
``solve_system`` bypasses Newton and solves the linear saddle-point system

.. math::

    (I - \alpha\nu A)\,\mathbf{U} - G\,\mathbf{1}
    = \nu A\,\mathbf{u}_\text{approx} + \mathbf{f}(t),

.. math::

    B\bigl(\mathbf{u}_\text{approx} + \alpha\,\mathbf{U}\bigr) = q(t),

directly by Schur-complement elimination:

1. :math:`\mathbf{r}_\text{eff} = \nu A\,\mathbf{u}_\text{approx} + \mathbf{f}(t)`.
2. Solve :math:`(I - \alpha\nu A)\,\mathbf{w} = \mathbf{r}_\text{eff}`.
3. Solve :math:`(I - \alpha\nu A)\,\mathbf{v}_0 = \mathbf{1}`.
4. :math:`G = \bigl(q(t) - B\,\mathbf{u}_\text{approx} - \alpha\,B\,\mathbf{w}\bigr)
           /\!\bigl(\alpha\,B\,\mathbf{v}_0\bigr)`.
5. Return :math:`\mathbf{U} = \mathbf{w} + G\,\mathbf{v}_0`, :math:`G`.

At the SDC fixed point this reproduces the DAE collocation solution.

Spatial discretisation
-----------------------
A **fourth-order** FD Laplacian pushes the spatial error floor to
:math:`\mathcal{O}(\Delta x^4) \approx 10^{-12}` with ``nvars = 1023``.
With homogeneous Dirichlet BCs the boundary-correction vector
:math:`b_\text{bc}` vanishes.

Classes
-------
stokes_poiseuille_1d_fd
    Problem class (inherits from
    :class:`~pySDC.projects.DAE.misc.problemDAE.ProblemDAE`).
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

from pySDC.projects.DAE.misc.problemDAE import ProblemDAE
from pySDC.projects.DAE.misc.meshDAE import MeshDAE


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
# Problem class
# ---------------------------------------------------------------------------

class stokes_poiseuille_1d_fd(ProblemDAE):
    r"""
    1-D unsteady Stokes / Poiseuille problem discretised with fourth-order
    finite differences.

    The state variable is a :class:`~pySDC.projects.DAE.misc.meshDAE.MeshDAE`
    with shape ``(2, nvars)`` (two components of length ``nvars`` each):

    * ``u.diff[:]`` – velocity on :math:`N` interior grid points.
    * ``u.alg[0]`` – pressure gradient :math:`G`; ``u.alg[1:]`` is unused.

    This class is compatible with the
    :class:`~pySDC.projects.DAE.sweepers.semiImplicitDAE.SemiImplicitDAE`
    sweeper, which expects ``eval_f(u, du, t)`` to return the fully-implicit
    DAE residual.

    **Exact solution** (manufactured):

    .. math::

        u_\text{ex}(y_i, t) = \sin(\pi y_i)\,\sin(t), \quad
        G_\text{ex}(t) = \cos(t).

    Parameters
    ----------
    nvars : int
        Number of interior grid points (default 127; must be ≥ 5).
    nu : float
        Kinematic viscosity :math:`\nu` (default 1.0).
    newton_tol : float
        Tolerance passed to ``ProblemDAE``; unused since ``solve_system``
        is overridden with a direct solver (default 1e-10).

    Attributes
    ----------
    dx : float
        Grid spacing :math:`1/(N+1)`.
    xvalues : numpy.ndarray
        Interior grid-point :math:`y`-coordinates.
    A : scipy.sparse.csc_matrix
        :math:`\nu`-scaled fourth-order FD Laplacian.
    ones : numpy.ndarray
        Vector of ones (pressure-gradient coupling, shape ``(nvars,)``).
    C_B : float
        :math:`h\sum_i \sin(\pi y_i)` – coefficient in
        :math:`q(t) = C_B\sin(t)`.
    """

    def __init__(self, nvars=127, nu=1.0, newton_tol=1e-10):
        if nvars < 5:
            raise ValueError(
                f'nvars must be >= 5 for the 4th-order FD Laplacian; got {nvars}'
            )
        # ProblemDAE: super().__init__((nvars, None, dtype)) → MeshDAE shape (2, nvars)
        super().__init__(nvars=nvars, newton_tol=newton_tol)
        self._makeAttributeAndRegister('nvars', 'nu', localVars=locals(), readOnly=True)

        self.dx = 1.0 / (nvars + 1)
        self.xvalues = np.linspace(self.dx, 1.0 - self.dx, nvars)

        # nu-scaled Laplacian (4th-order FD, zero BCs → no b_bc correction)
        self.A = nu * _build_laplacian(nvars, self.dx)

        # Discrete-integral operator B = h * 1^T (used as a plain vector)
        self.ones = np.ones(nvars)
        self.C_B = self.dx * float(np.sum(np.sin(np.pi * self.xvalues)))

    # -----------------------------------------------------------------------
    # Manufactured-solution helpers
    # -----------------------------------------------------------------------

    def _q(self, t):
        r"""
        Flow-rate constraint RHS: :math:`q(t) = C_B\,\sin(t)`.

        Parameters
        ----------
        t : float

        Returns
        -------
        float
        """
        return self.C_B * np.sin(t)

    def _forcing(self, t):
        r"""
        Manufactured forcing consistent with :math:`u_\text{ex}` and
        :math:`G_\text{ex} = \cos(t)`:

        .. math::

            f(y, t) = \sin(\pi y)\cos(t)
                    + \nu\pi^2\sin(\pi y)\sin(t) - \cos(t).

        Parameters
        ----------
        t : float

        Returns
        -------
        numpy.ndarray, shape (nvars,)
        """
        y = self.xvalues
        return (
            np.sin(np.pi * y) * np.cos(t)
            + self.nu * np.pi**2 * np.sin(np.pi * y) * np.sin(t)
            - np.cos(t) * self.ones
        )

    # -----------------------------------------------------------------------
    # pySDC / SemiImplicitDAE interface
    # -----------------------------------------------------------------------

    def eval_f(self, u, du, t):
        r"""
        Evaluate the fully-implicit DAE residual
        :math:`F(\mathbf{u}, \mathbf{u}', t) = 0`:

        .. math::

            F_\text{diff} = \mathbf{u}' - \nu A\,\mathbf{u}
                          - G\,\mathbf{1} - \mathbf{f}(t),

        .. math::

            F_\text{alg} = B\,\mathbf{u} - q(t).

        Parameters
        ----------
        u : MeshDAE
            Current state.  ``u.diff[:]`` = velocity, ``u.alg[0]`` = G.
        du : MeshDAE
            Current derivative.  ``du.diff[:]`` = :math:`\mathbf{u}'`.
        t : float
            Current time.

        Returns
        -------
        f : MeshDAE
            Residual: ``f.diff[:]`` = :math:`F_\text{diff}`,
            ``f.alg[0]`` = :math:`F_\text{alg}` (scalar; zero at the
            exact solution).
        """
        f = self.dtype_f(self.init, val=0.0)
        u_vel = np.asarray(u.diff)
        du_vel = np.asarray(du.diff)
        G = float(u.alg[0])

        f.diff[:] = du_vel - (self.A.dot(u_vel) + G * self.ones + self._forcing(t))
        f.alg[0] = self.dx * float(np.sum(u_vel)) - self._q(t)

        self.work_counters['rhs']()
        return f

    def solve_system(self, impl_sys, u_approx, factor, u0, t):
        r"""
        Direct Schur-complement solve for one SDC node (bypasses Newton).

        The ``SemiImplicitDAE`` sweeper calls this method to find
        :math:`(\mathbf{U}_m, G_m)` satisfying

        .. math::

            \mathbf{U}_m - \bigl(\nu A\,(\mathbf{u}_\text{approx}
            + \alpha\,\mathbf{U}_m) + G_m\,\mathbf{1} + \mathbf{f}(t)\bigr) = 0,

        .. math::

            B\,(\mathbf{u}_\text{approx} + \alpha\,\mathbf{U}_m) = q(t),

        which reduces to the system shown in the module docstring.

        After the solve:

        * ``me.diff[:]`` = :math:`\mathbf{U}_m` (velocity derivative at
          the node, stored in ``L.f[m].diff`` by the sweeper).
        * ``me.alg[0]`` = :math:`G_m` (stored in ``L.u[m].alg`` by the
          sweeper).

        Parameters
        ----------
        impl_sys : callable
            The :func:`~pySDC.projects.DAE.sweepers.semiImplicitDAE.SemiImplicitDAE.F`
            function (unused; linear system solved directly).
        u_approx : MeshDAE
            Current approximation of the velocity at the node.
        factor : float
            Implicit step-size prefactor :math:`\alpha`.
        u0 : MeshDAE
            Initial guess (unused; direct solver).
        t : float
            Current time.

        Returns
        -------
        me : MeshDAE
            Contains :math:`(\mathbf{U}_m, G_m)`.
        """
        me = self.dtype_u(self.init, val=0.0)
        u_vel_approx = np.asarray(u_approx.diff).copy()

        # Effective RHS for the velocity block.
        rhs_eff = self.A.dot(u_vel_approx) + self._forcing(t)

        # Assemble operator M = I - alpha * nu * A.
        M = sp.eye(self.nvars, format='csc') - factor * self.A

        # Step 1: solve without pressure.
        w = spsolve(M, rhs_eff)

        # Step 2: unit-pressure response.
        v0 = spsolve(M, self.ones)

        # Step 3: enforce constraint  B*(u_approx + alpha*(w + G*v0)) = q(t).
        Bw = self.dx * float(np.sum(w))
        Bv0 = self.dx * float(np.sum(v0))          # > 0 for alpha > 0
        Bu_approx = self.dx * float(np.sum(u_vel_approx))
        G_new = (self._q(t) - Bu_approx - factor * Bw) / (factor * Bv0)

        # Step 4: assemble velocity derivative and algebraic variable.
        me.diff[:] = w + G_new * v0
        me.alg[0] = G_new
        return me

    def u_exact(self, t):
        r"""
        Exact solution at time :math:`t`.

        Parameters
        ----------
        t : float

        Returns
        -------
        me : MeshDAE
            ``me.diff[:]`` = :math:`\sin(\pi y_i)\sin(t)`,
            ``me.alg[0]`` = :math:`\cos(t)`.
        """
        me = self.dtype_u(self.init, val=0.0)
        me.diff[:] = np.sin(np.pi * self.xvalues) * np.sin(t)
        me.alg[0] = np.cos(t)
        return me

    def du_exact(self, t):
        r"""
        Exact time derivative at time :math:`t`.

        Parameters
        ----------
        t : float

        Returns
        -------
        me : MeshDAE
            ``me.diff[:]`` = :math:`\sin(\pi y_i)\cos(t)`,
            ``me.alg[0]`` = :math:`-\sin(t)`.
        """
        me = self.dtype_u(self.init, val=0.0)
        me.diff[:] = np.sin(np.pi * self.xvalues) * np.cos(t)
        me.alg[0] = -np.sin(t)
        return me
