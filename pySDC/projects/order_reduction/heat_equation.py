import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

from pySDC.core.problem import Problem
from pySDC.implementations.datatype_classes.mesh import mesh


class HeatEquation_1D_FD_homogeneous_Dirichlet(Problem):
    r"""
    Unforced 1D heat equation with homogeneous Dirichlet boundary conditions.

    Solves the initial-value problem

    .. math::
        \frac{\partial u}{\partial t} = \nu \frac{\partial^2 u}{\partial x^2},
        \quad x \in (0, 1),\quad u(0, t) = u(1, t) = 0.

    The exact solution is

    .. math::
        u(x, t) = \sin\!\left(\pi k x\right) \exp\!\left(-\nu \rho_{\mathrm{FD}} t\right),

    where :math:`\rho_{\mathrm{FD}} = \nu(2 - 2\cos(\pi k \,\Delta x))/\Delta x^2` is the
    *discrete* eigenvalue magnitude of the FD Laplacian for the sine mode of frequency :math:`k`.
    This ensures that :math:`u_{\mathrm{exact}}` is the **exact** solution of the
    *discrete* method-of-lines ODE (no spatial error), which makes it a clean benchmark
    for measuring purely temporal convergence.

    Because the boundary values are identically zero, there is no time-dependent
    boundary contribution to the right-hand side, and SDC achieves the expected
    full temporal order.

    Parameters
    ----------
    nvars : int, optional
        Number of interior degrees of freedom (default 127).
    nu : float, optional
        Diffusion coefficient :math:`\nu` (default 0.1).
    freq : int, optional
        Spatial frequency :math:`k` (default 1).
    """

    dtype_u = mesh
    dtype_f = mesh

    def __init__(self, nvars=127, nu=0.1, freq=1):
        super().__init__(init=(nvars, None, np.dtype('float64')))

        dx = 1.0 / (nvars + 1)
        xvalues = np.array([dx * (i + 1) for i in range(nvars)])

        # Second-order central-difference Laplacian with homogeneous Dirichlet BCs
        diagonals = [np.ones(nvars - 1), -2.0 * np.ones(nvars), np.ones(nvars - 1)]
        A = sp.diags(diagonals, [-1, 0, 1], shape=(nvars, nvars), format='csc')
        A *= nu / dx**2

        # Discrete eigenvalue for the sine mode (magnitude; A already contains nu)
        # eigenvalue = A * sine_mode / sine_mode  (negative; rho_disc = |eigenvalue|)
        sine_mode = np.sin(np.pi * freq * xvalues)
        self.rho_disc = float(-A.dot(sine_mode)[nvars // 2] / sine_mode[nvars // 2])

        self.A = A
        self.dx = dx
        self.xvalues = xvalues
        self.Id = sp.eye(nvars, format='csc')
        self._makeAttributeAndRegister('nvars', 'nu', 'freq', localVars=locals(), readOnly=True)

    def eval_f(self, u, t):
        """
        Evaluate the right-hand side :math:`f(u) = \\nu A u`.

        Parameters
        ----------
        u : dtype_u
            Current numerical solution.
        t : float
            Current time (unused).

        Returns
        -------
        f : dtype_f
            Right-hand side.
        """
        f = self.f_init
        f[:] = self.A.dot(u.flatten())
        return f

    def solve_system(self, rhs, factor, u0, t):
        r"""
        Solve :math:`(I - \text{factor} \cdot A)\,u = \text{rhs}`.

        Parameters
        ----------
        rhs : dtype_u
            Right-hand side.
        factor : float
            Implicit prefactor.
        u0 : dtype_u
            Initial guess (unused).
        t : float
            Current time (unused).

        Returns
        -------
        sol : dtype_u
            Solution.
        """
        sol = self.u_init
        sol[:] = spsolve(self.Id - factor * self.A, rhs.flatten())
        return sol

    def u_exact(self, t, **kwargs):
        r"""
        Exact solution :math:`u(x,t) = \sin(\pi k x)\,\exp(-\nu\rho_{\mathrm{FD}}\,t)`.

        Parameters
        ----------
        t : float
            Evaluation time.

        Returns
        -------
        sol : dtype_u
            Exact solution values at the interior grid points.
        """
        sol = self.u_init
        sol[:] = np.sin(np.pi * self.freq * self.xvalues) * np.exp(-self.rho_disc * t)
        return sol


class HeatEquation_1D_FD_time_dependent_Dirichlet(Problem):
    r"""
    Unforced 1D heat equation with time-dependent Dirichlet boundary conditions.

    Solves the initial-value problem

    .. math::
        \frac{\partial u}{\partial t} = \nu \frac{\partial^2 u}{\partial x^2},
        \quad x \in (0, 1),

    with boundary conditions :math:`u(0,t) = g_0(t)` and :math:`u(1,t) = g_1(t)` derived
    from the continuous exact solution

    .. math::
        u(x, t) = \cos(\pi k x)\,\exp\!\left(-\nu (\pi k)^2 t\right).

    The method-of-lines ODE for the interior grid points is

    .. math::
        \mathbf{u}' = A_0 \mathbf{u} + \mathbf{b}(t),

    where :math:`A_0` is the second-order FD Laplacian matrix with *zero* Dirichlet BCs
    and the boundary correction vector is

    .. math::
        \mathbf{b}(t)_j = \begin{cases}
            \dfrac{\nu}{\Delta x^2}\,g_0(t) & j = 0, \\[4pt]
            \dfrac{\nu}{\Delta x^2}\,g_1(t) & j = N-1, \\[4pt]
            0 & \text{otherwise.}
        \end{cases}

    The ``solve_system`` method of this class **intentionally omits** the diagonal
    implicit contribution :math:`\alpha \mathbf{b}(t)` from the linear solve.  This is
    the *naive* implementation that one naturally arrives at when setting up a fully
    implicit SDC solver without special treatment of the stiff boundary correction, and
    it leads to **order reduction**: the effective temporal convergence order drops from
    the theoretical SDC order to 1 (or close to 1).

    See :class:`HeatEquation_1D_FD_time_dependent_Dirichlet_full` for the corrected
    version that achieves full SDC order.

    Parameters
    ----------
    nvars : int, optional
        Number of interior degrees of freedom (default 127).
    nu : float, optional
        Diffusion coefficient :math:`\nu` (default 0.1).
    freq : int, optional
        Spatial frequency :math:`k` (default 1).
    """

    dtype_u = mesh
    dtype_f = mesh

    def __init__(self, nvars=127, nu=0.1, freq=1):
        super().__init__(init=(nvars, None, np.dtype('float64')))

        dx = 1.0 / (nvars + 1)
        xvalues = np.array([dx * (i + 1) for i in range(nvars)])

        # FD Laplacian with zero Dirichlet BCs
        diagonals = [np.ones(nvars - 1), -2.0 * np.ones(nvars), np.ones(nvars - 1)]
        A = sp.diags(diagonals, [-1, 0, 1], shape=(nvars, nvars), format='csc')
        A *= nu / dx**2

        # Boundary values at x=0 and x=1
        g0 = float(np.cos(np.pi * freq * 0.0))  # = 1.0 for all freq
        g1 = float(np.cos(np.pi * freq * 1.0))  # = cos(pi*freq)

        # Precompute the static part of the BC correction (time factor is exp(-nu*(pi*freq)^2 * t))
        bc_static = np.zeros(nvars)
        bc_static[0] = nu / dx**2 * g0
        bc_static[-1] = nu / dx**2 * g1

        self.A = A
        self.dx = dx
        self.xvalues = xvalues
        self.Id = sp.eye(nvars, format='csc')
        self.bc_static = bc_static
        self._makeAttributeAndRegister('nvars', 'nu', 'freq', localVars=locals(), readOnly=True)

    def _bc_correction(self, t):
        """
        Time-dependent boundary correction :math:`\\mathbf{b}(t)`.

        Parameters
        ----------
        t : float
            Current time.

        Returns
        -------
        b : np.ndarray
            BC correction vector (shape ``(nvars,)``).
        """
        decay = np.exp(-self.nu * (np.pi * self.freq) ** 2 * t)
        return self.bc_static * decay

    def eval_f(self, u, t):
        """
        Evaluate :math:`f(u, t) = A_0 u + \\mathbf{b}(t)`.

        Parameters
        ----------
        u : dtype_u
            Current numerical solution.
        t : float
            Current time.

        Returns
        -------
        f : dtype_f
            Right-hand side including the BC correction.
        """
        f = self.f_init
        f[:] = self.A.dot(u.flatten()) + self._bc_correction(t)
        return f

    def solve_system(self, rhs, factor, u0, t):
        r"""
        Solve :math:`(I - \text{factor} \cdot A_0)\,u = \text{rhs}`.

        .. note::
            The boundary correction :math:`\text{factor} \cdot \mathbf{b}(t)` is **not**
            added to the right-hand side here.  This omission is deliberate: it
            demonstrates the **order reduction** that occurs when the stiff BC term is
            not treated implicitly.  See
            :class:`HeatEquation_1D_FD_time_dependent_Dirichlet_full` for the corrected
            version.

        Parameters
        ----------
        rhs : dtype_u
            Right-hand side.
        factor : float
            Implicit prefactor.
        u0 : dtype_u
            Initial guess (unused).
        t : float
            Current time.

        Returns
        -------
        sol : dtype_u
            Solution.
        """
        sol = self.u_init
        # Intentionally omits factor * self._bc_correction(t) to demonstrate order reduction
        sol[:] = spsolve(self.Id - factor * self.A, rhs.flatten())
        return sol

    def u_exact(self, t, **kwargs):
        r"""
        Exact solution :math:`u(x,t) = \cos(\pi k x)\,\exp(-\nu(\pi k)^2 t)`.

        Parameters
        ----------
        t : float
            Evaluation time.

        Returns
        -------
        sol : dtype_u
            Exact solution at interior grid points.
        """
        sol = self.u_init
        sol[:] = np.cos(np.pi * self.freq * self.xvalues) * np.exp(-self.nu * (np.pi * self.freq) ** 2 * t)
        return sol


class HeatEquation_1D_FD_time_dependent_Dirichlet_full(HeatEquation_1D_FD_time_dependent_Dirichlet):
    r"""
    Corrected version of :class:`HeatEquation_1D_FD_time_dependent_Dirichlet`.

    The ``solve_system`` method adds :math:`\text{factor} \cdot \mathbf{b}(t)` to the
    right-hand side before solving, thereby treating the stiff boundary correction
    implicitly.  This recovers the **full** SDC temporal order of convergence.

    Parameters
    ----------
    nvars : int, optional
        Number of interior degrees of freedom (default 127).
    nu : float, optional
        Diffusion coefficient :math:`\nu` (default 0.1).
    freq : int, optional
        Spatial frequency :math:`k` (default 1).
    """

    def solve_system(self, rhs, factor, u0, t):
        r"""
        Solve :math:`(I - \text{factor} \cdot A_0)\,u = \text{rhs} + \text{factor}\,\mathbf{b}(t)`.

        Including the boundary correction in the solve restores full SDC order.

        Parameters
        ----------
        rhs : dtype_u
            Right-hand side.
        factor : float
            Implicit prefactor.
        u0 : dtype_u
            Initial guess (unused).
        t : float
            Current time.

        Returns
        -------
        sol : dtype_u
            Solution.
        """
        sol = self.u_init
        corrected_rhs = rhs.flatten() + factor * self._bc_correction(t)
        sol[:] = spsolve(self.Id - factor * self.A, corrected_rhs)
        return sol
