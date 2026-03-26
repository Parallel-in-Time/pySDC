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

where :math:`G(t)` is the unknown (spatially uniform) pressure gradient and
:math:`q(t)` is a prescribed flow-rate.  After finite-difference discretisation
this becomes the index-1 semi-explicit DAE

.. math::

    \mathbf{u}' = \nu A\,\mathbf{u} + G\,\mathbf{1} + \mathbf{f}(t),

.. math::

    0 = B\,\mathbf{u} - q(t),

where :math:`A` is the FD Laplacian,
:math:`B = h\,\mathbf{1}^T` (rectangle-rule integral, :math:`h = 1/(N+1)`),
:math:`\mathbf{1}` is the vector of ones, and :math:`G(t)` is the Lagrange
multiplier that enforces the flow-rate constraint.

Manufactured solution
---------------------
.. math::

    u_\text{ex}(y, t) = \sin(\pi y)\,\sin(t), \quad
    G_\text{ex}(t)    = \cos(t).

The corresponding manufactured forcing that makes the system consistent is

.. math::

    f(y, t) = \sin(\pi y)\cos(t) + \nu\pi^2\sin(\pi y)\sin(t) - \cos(t).

IMEX split and saddle-point solve
----------------------------------
The IMEX split used with the ``imex_1st_order`` sweeper is:

* :math:`f_\text{impl} = \nu A\,\mathbf{u}` – autonomous diffusion (stiff).
* :math:`f_\text{expl} = G\,\mathbf{1} + \mathbf{f}(t)` – pressure gradient
  and manufactured forcing (non-stiff).

The pressure :math:`G` is the algebraic variable of the DAE.  It is updated
at each SDC node during ``solve_system`` via a **Schur-complement saddle-point
solve**:

1. Solve :math:`(I - \alpha\nu A)\,\mathbf{w} = \mathbf{r}` (standard diffusion
   solve).
2. Solve :math:`(I - \alpha\nu A)\,\mathbf{v} = \mathbf{1}` (unit-load solve).
3. Enforce the constraint:
   :math:`G = \bigl(q(t) - B\,\mathbf{w}\bigr) /\! \bigl(\alpha\,B\,\mathbf{v}\bigr)`.
4. Return :math:`\mathbf{u} = \mathbf{w} + \alpha\,G\,\mathbf{v}`.

Between successive SDC nodes :math:`G` is stored in ``self._G_last`` and used
in the subsequent ``eval_f`` call.

Spatial discretisation
-----------------------
A **fourth-order** FD Laplacian is used to push the spatial error floor to
:math:`\mathcal{O}(\Delta x^4) \approx 10^{-12}` with ``nvars = 1023``.
Because the exact velocity profile satisfies homogeneous Dirichlet BCs no
boundary-correction vector :math:`b_\text{bc}` is needed.

Classes
-------
stokes_poiseuille_1d_fd
    Single-variant problem class.
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

from pySDC.core.problem import Problem, WorkCounter
from pySDC.implementations.datatype_classes.mesh import mesh, imex_mesh


# ---------------------------------------------------------------------------
# Fourth-order FD Laplacian (same stencil as AllenCahn_1D_FD)
# ---------------------------------------------------------------------------

def _build_laplacian(n, dx):
    r"""
    Assemble the fourth-order FD Laplacian on *n* interior points with
    **zero** Dirichlet boundary conditions.

    Interior rows use the centred stencil
    :math:`(-u_{k-2}+16u_{k-1}-30u_k+16u_{k+1}-u_{k+2})/(12\Delta x^2)`;
    the outermost rows use a 6-point one-sided stencil to maintain
    fourth-order accuracy without ghost points.

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

    # Row 0: 6-point one-sided stencil (u_L = 0 → no boundary correction).
    L[0, 0] = -15
    L[0, 1] = -4
    L[0, 2] = 14
    L[0, 3] = -6
    L[0, 4] = 1

    # Row 1: standard centred stencil; u_{-1} = u_L = 0.
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

    # Row n-2: standard centred stencil; u_n = u_R = 0.
    L[n - 2, n - 4] = -1
    L[n - 2, n - 3] = 16
    L[n - 2, n - 2] = -30
    L[n - 2, n - 1] = 16

    # Row n-1: mirror of row 0 (u_R = 0 → no boundary correction).
    L[n - 1, n - 5] = 1
    L[n - 1, n - 4] = -6
    L[n - 1, n - 3] = 14
    L[n - 1, n - 2] = -4
    L[n - 1, n - 1] = -15

    return (L * inv12dx2).tocsc()


# ---------------------------------------------------------------------------
# Problem class
# ---------------------------------------------------------------------------

class stokes_poiseuille_1d_fd(Problem):
    r"""
    1-D unsteady Stokes / Poiseuille problem discretised with fourth-order
    finite differences.

    **State variable**: velocity :math:`\mathbf{u} \in \mathbb{R}^N`
    (N interior grid points, zero Dirichlet BCs).

    **Algebraic variable**: pressure gradient :math:`G(t) \in \mathbb{R}`
    (Lagrange multiplier for the flow-rate constraint).

    **Exact solution** (manufactured):

    .. math::

        u_\text{ex}(y_i, t) = \sin(\pi y_i)\,\sin(t), \quad
        G_\text{ex}(t) = \cos(t).

    **Flow-rate constraint RHS**:

    .. math::

        q(t) = B\,\mathbf{u}_\text{ex}(t) = C_B\,\sin(t),
        \quad C_B = h\sum_i \sin(\pi y_i).

    Parameters
    ----------
    nvars : int
        Number of interior grid points (default 127; must be ≥ 5).
    nu : float
        Kinematic viscosity :math:`\nu` (default 1.0).

    Attributes
    ----------
    dx : float
        Grid spacing :math:`1/(N+1)`.
    xvalues : numpy.ndarray
        Interior grid-point :math:`y`-coordinates.
    A : scipy.sparse.csc_matrix
        :math:`\nu`-scaled fourth-order FD Laplacian.
    ones : numpy.ndarray
        Vector of ones (pressure-gradient coupling).
    C_B : float
        :math:`h\sum_i \sin(\pi y_i)` – coefficient in :math:`q(t)`.
    s : float
        :math:`h N` – discrete integral of the unit vector (:math:`B\mathbf{1}`).
    _G_last : float
        Most recently computed algebraic variable :math:`G`.
    """

    dtype_u = mesh
    dtype_f = imex_mesh

    def __init__(self, nvars=127, nu=1.0):
        if nvars < 5:
            raise ValueError(f'nvars must be >= 5 for the 4th-order FD Laplacian; got {nvars}')
        super().__init__((nvars, None, np.dtype('float64')))
        self._makeAttributeAndRegister('nvars', 'nu', localVars=locals(), readOnly=True)

        self.dx = 1.0 / (nvars + 1)
        self.xvalues = np.linspace(self.dx, 1.0 - self.dx, nvars)

        # nu-scaled Laplacian (4th-order FD, zero BCs → no b_bc correction)
        self.A = nu * _build_laplacian(nvars, self.dx)

        # Discrete-integral operator B = h * 1^T as a plain vector
        self.ones = np.ones(nvars)
        self.s = self.dx * nvars                              # B * 1 = h*N
        self.C_B = self.dx * np.sum(np.sin(np.pi * self.xvalues))  # B * sin(pi*y)

        # Initialise algebraic variable to G_ex(0) = cos(0) = 1
        self._G_last = 1.0

        self.work_counters['rhs'] = WorkCounter()

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
        Manufactured forcing :math:`f(y_i, t)` that makes the system
        consistent with :math:`u_\text{ex}` and :math:`G_\text{ex}`:

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
    # pySDC interface
    # -----------------------------------------------------------------------

    def eval_f(self, u, t):
        r"""
        Evaluate the IMEX right-hand side.

        * :math:`f_\text{impl} = \nu A\,\mathbf{u}` (stiff diffusion).
        * :math:`f_\text{expl} = G\,\mathbf{1} + \mathbf{f}(t)` where
          :math:`G` is taken from ``self._G_last`` (set by ``solve_system``
          immediately before this call, or initialised to :math:`G_\text{ex}(0)`).

        Parameters
        ----------
        u : dtype_u
            Current velocity.
        t : float
            Current time.

        Returns
        -------
        f : imex_mesh
        """
        f = self.dtype_f(self.init)
        f.impl[:] = self.A.dot(np.asarray(u))
        f.expl[:] = self._G_last * self.ones + self._forcing(t)
        self.work_counters['rhs']()
        return f

    def solve_system(self, rhs, factor, u0, t):
        r"""
        Solve the saddle-point system via Schur-complement elimination.

        Given the SDC right-hand side :math:`\mathbf{r}` and step-size
        prefactor :math:`\alpha = \Delta t\,\tilde{q}_{mm}`, find
        :math:`(\mathbf{u}, G)` satisfying

        .. math::

            (I - \alpha\nu A)\,\mathbf{u} = \mathbf{r} + \alpha\,G\,\mathbf{1},

        .. math::

            B\,\mathbf{u} = q(t).

        **Algorithm** (Schur complement):

        1. :math:`\mathbf{w} = (I - \alpha\nu A)^{-1}\,\mathbf{r}`.
        2. :math:`\mathbf{v} = (I - \alpha\nu A)^{-1}\,\mathbf{1}`.
        3. :math:`G = \bigl(q(t) - B\,\mathbf{w}\bigr) /
                      \bigl(\alpha\,B\,\mathbf{v}\bigr)`.
        4. :math:`\mathbf{u} = \mathbf{w} + \alpha\,G\,\mathbf{v}`.

        The computed :math:`G` is stored in ``self._G_last`` for use in the
        immediately following ``eval_f`` call.

        Parameters
        ----------
        rhs : dtype_u
            SDC right-hand side.
        factor : float
            Implicit step-size prefactor :math:`\alpha`.
        u0 : dtype_u
            Initial guess (unused; direct solver).
        t : float
            Current time.

        Returns
        -------
        me : dtype_u
            Updated velocity satisfying :math:`B\,\mathbf{u} = q(t)`.
        """
        me = self.dtype_u(self.init)
        rhs_np = np.asarray(rhs)

        if factor == 0.0:
            # Trivial step: no diffusion, just copy rhs.
            # Enforce constraint by computing the required G correction.
            Br = self.dx * np.sum(rhs_np)
            q_t = self._q(t)
            # With factor=0 there is no v to scale, so we handle
            # the constraint by projecting rhs onto the constraint manifold.
            G_corr = (q_t - Br) / self.s
            me[:] = rhs_np + G_corr * self.ones
            self._G_last = G_corr
            return me

        # Assemble and factor the diffusion operator.
        M = sp.eye(self.nvars, format='csc') - factor * self.A

        # Step 1: diffusion-only solve.
        w = spsolve(M, rhs_np)

        # Step 2: unit-load solve (effect of a unit G contribution).
        v = spsolve(M, self.ones)

        # Step 3: enforce flow-rate constraint  B*(w + factor*G*v) = q(t).
        Bw = self.dx * np.sum(w)          # B.dot(w) = h * sum(w)
        Bv = self.dx * np.sum(v)          # B.dot(v) = h * sum(v) > 0
        G_new = (self._q(t) - Bw) / (factor * Bv)

        # Step 4: assemble velocity.
        me[:] = w + factor * G_new * v

        # Persist G for the next eval_f call.
        self._G_last = float(G_new)
        return me

    def u_exact(self, t):
        r"""
        Exact velocity :math:`u_\text{ex}(y_i, t) = \sin(\pi y_i)\,\sin(t)`.

        Parameters
        ----------
        t : float

        Returns
        -------
        me : dtype_u
        """
        me = self.dtype_u(self.init, val=0.0)
        me[:] = np.sin(np.pi * self.xvalues) * np.sin(t)
        return me

    @staticmethod
    def G_exact(t):
        r"""
        Exact pressure gradient :math:`G_\text{ex}(t) = \cos(t)`.

        Parameters
        ----------
        t : float

        Returns
        -------
        float
        """
        return float(np.cos(t))
