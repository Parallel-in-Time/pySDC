r"""
Stochastic differential equations in Stratonovich form,

.. math::

    dX_t = \underline{a}(t, X_t)\, dt + \sum_{r=1}^{R} b_r(t, X_t) \circ dW^r_t,

solved for a whole ensemble of Brownian paths at once: the solution ``u`` has shape ``(n_samples, ndim)``, one row
per path. The Itô form :math:`dX_t = a\, dt + \sum_r b_r\, dW^r_t` has the drift
:math:`a = \underline{a} + \frac{1}{2} \sum_r (b_r \cdot \nabla) b_r`.

On each time step, SDC does not see the SDE but the random ODE (SBB-ODE)

.. math::

    \frac{dx}{dt} = \underline{a}(t, x) + \sum_{r=1}^{R} b_r(t, x) \frac{dB_r^{(J)}}{dt}(t),

with the smooth Brownian bridges :math:`B_r^{(J)}` of :mod:`pySDC.projects.SDC4SDEs.brownian`. The sweeper calls
:meth:`SDEProblem.set_step` before it starts on a step, and the right-hand side returned by :meth:`SDEProblem.eval_f`
keeps the drift and the diffusion apart, so that the sweeper can integrate the oscillating part of the bridge with
its own quadrature.
"""

import numpy as np
from scipy.linalg import expm

from pySDC.core.problem import Problem, WorkCounter
from pySDC.implementations.datatype_classes.mesh import mesh
from pySDC.projects.SDC4SDEs.brownian import bridge_coefficients


class SDEProblem(Problem):
    r"""
    Base class of the SDE problems, driven by given Brownian paths.

    Subclasses set ``ndim`` (dimension of the state) and ``n_noise`` (number R of Wiener processes), and implement
    :meth:`drift`, :meth:`diffusion`, :meth:`diffusion_derivative` and :meth:`u_exact`.

    The right-hand side returned by :meth:`eval_f` is a mesh of shape ``(1 + R, n_samples, ndim)``: the Stratonovich
    drift first, then the R diffusion vectors :math:`b_r`. :meth:`rhs` combines them with the derivative of the
    bridge.

    Work is counted per evaluation of the whole ensemble, so that it is the work per sample: ``drift`` counts
    evaluations of the drift (Itô or Stratonovich), ``diffusion`` evaluations of all R diffusion vectors, and
    ``diffusion_derivative`` evaluations of their derivatives (for Milstein).

    Args:
        brownian_path (numpy.ndarray): paths of shape (R, n_samples, n_fine + 1) on a uniform grid of [0, T],
            starting at zero, see :func:`pySDC.projects.SDC4SDEs.brownian.brownian_paths`
        T (float): end time of the paths
        n_bridge_terms (int): number J of sine terms of the bridge beyond the linear one
    """

    dtype_u = mesh
    dtype_f = mesh
    ndim = 1
    n_noise = 1

    def __init__(self, brownian_path, T=1.0, n_bridge_terms=0):
        if brownian_path.ndim != 3 or brownian_path.shape[0] != self.n_noise:
            raise ValueError(f'expected paths of shape ({self.n_noise}, n_samples, n_fine + 1)')
        n_samples = brownian_path.shape[1]
        super().__init__(init=((n_samples, self.ndim), None, np.dtype('float64')))
        self._makeAttributeAndRegister('brownian_path', 'T', 'n_bridge_terms', localVars=locals(), readOnly=True)
        self.n_samples = n_samples
        self.dt_fine = T / (brownian_path.shape[2] - 1)
        self.init_f = ((1 + self.n_noise, n_samples, self.ndim), None, np.dtype('float64'))
        for key in ['drift', 'diffusion', 'diffusion_derivative']:
            self.work_counters[key] = WorkCounter()
        self.set_step(0.0, T)

    def _fine_index(self, t):
        """Index of time t on the grid of the paths, which t has to be on."""
        i = round(t / self.dt_fine)
        if abs(i * self.dt_fine - t) > 1e-9 * max(1.0, abs(t)):
            raise ValueError(f'time {t} is not on the grid of the Brownian paths (spacing {self.dt_fine})')
        return i

    def W(self, t):
        """Values of the Brownian paths at time t, shape (R, n_samples)."""
        return self.brownian_path[..., self._fine_index(t)]

    def set_step(self, t0, dt):
        """
        Compute the smooth Brownian bridges on the step [t0, t0 + dt], whose ends have to be on the grid of the paths.

        Sets ``dW`` (increments of the paths over the step), ``eta0`` and ``xi`` (coefficients of the bridge).
        """
        i0, i1 = self._fine_index(t0), self._fine_index(t0 + dt)
        W = self.brownian_path[..., i0 : i1 + 1]
        self.step = (t0, dt)
        self.dW = W[..., -1] - W[..., 0]
        self.eta0, self.xi = bridge_coefficients(W, dt, self.n_bridge_terms)

    def dB(self, t):
        """Derivative of the smooth Brownian bridges at time t of the current step, shape (R, n_samples)."""
        t0, dt = self.step
        j = np.arange(1, self.n_bridge_terms + 1)
        return (self.eta0 + np.sqrt(2.0) * self.xi @ np.cos(j * np.pi * (t - t0) / dt)) / np.sqrt(dt)

    def eval_f(self, u, t):
        """Stratonovich drift and diffusion vectors, stacked to shape (1 + R, n_samples, ndim)."""
        f = mesh(self.init_f)
        f[0] = self.drift(u, t)
        f[1:] = self.diffusion(u, t)
        self.work_counters['drift']()
        self.work_counters['diffusion']()
        return f

    def rhs(self, f, t):
        """Right-hand side of the SBB-ODE at time t of the current step, from the output of eval_f."""
        return f[0] + (self.dB(t)[..., None] * f[1:]).sum(axis=0)

    def ito_drift(self, u, t):
        r"""Drift of the Itô form, :math:`\underline{a} + \frac{1}{2} \sum_r (b_r \cdot \nabla) b_r`."""
        self.work_counters['drift']()
        return self.drift(u, t) + 0.5 * np.trace(self.diffusion_derivative(u, t), axis1=0, axis2=1)

    def drift(self, u, t):
        r"""Stratonovich drift :math:`\underline{a}(t, u)`, shape (n_samples, ndim)."""
        raise NotImplementedError

    def diffusion(self, u, t):
        """Diffusion vectors b_r(t, u), shape (R, n_samples, ndim)."""
        raise NotImplementedError

    def diffusion_derivative(self, u, t):
        r"""Derivatives :math:`(b_s \cdot \nabla) b_r` at index [r, s], shape (R, R, n_samples, ndim)."""
        raise NotImplementedError

    def u_exact(self, t):
        """Exact solution of the SDE for the given Brownian paths at time t, shape (n_samples, ndim)."""
        raise NotImplementedError


class LinearScalarSDE(SDEProblem):
    r"""
    Geometric Brownian motion :math:`dX = (a - \frac{1}{2} b^2) X\, dt + b X \circ dW`, with the exact solution
    :math:`X_t = X_0 \exp((a - \frac{1}{2} b^2) t + b W_t)`.
    """

    def __init__(self, brownian_path, T=1.0, n_bridge_terms=0, a=0.5, b=1.0, u0=1.0):
        super().__init__(brownian_path, T, n_bridge_terms)
        self._makeAttributeAndRegister('a', 'b', 'u0', localVars=locals(), readOnly=True)

    def drift(self, u, t):
        return (self.a - 0.5 * self.b**2) * u

    def diffusion(self, u, t):
        return self.b * u[None]

    def diffusion_derivative(self, u, t):
        return self.b**2 * u[None, None]

    def u_exact(self, t):
        u = mesh(self.init)
        u[:, 0] = self.u0 * np.exp((self.a - 0.5 * self.b**2) * t + self.b * self.W(t)[0])
        return u


class NonlinearScalarSDE(SDEProblem):
    r"""
    The SDE :math:`dX = -a (1 - X^2)\, dt + b (1 - X^2) \circ dW` from Tian and Burrage, with the exact solution

    .. math::

        X_t = \frac{(1 + X_0) e^{-2 a t + 2 b W_t} + X_0 - 1}{(1 + X_0) e^{-2 a t + 2 b W_t} - X_0 + 1}.
    """

    def __init__(self, brownian_path, T=1.0, n_bridge_terms=0, a=1.0, b=1.0, u0=0.0):
        super().__init__(brownian_path, T, n_bridge_terms)
        self._makeAttributeAndRegister('a', 'b', 'u0', localVars=locals(), readOnly=True)

    def drift(self, u, t):
        return -self.a * (1 - u**2)

    def diffusion(self, u, t):
        return self.b * (1 - u[None] ** 2)

    def diffusion_derivative(self, u, t):
        return (self.b * (1 - u**2) * (-2 * self.b * u))[None, None]

    def u_exact(self, t):
        e = (1 + self.u0) * np.exp(-2 * self.a * t + 2 * self.b * self.W(t)[0])
        u = mesh(self.init)
        u[:, 0] = (e + self.u0 - 1) / (e - self.u0 + 1)
        return u


class LinearSystemSDE(SDEProblem):
    r"""
    The linear system :math:`dX = (A - \frac{1}{2}(B_1^2 + B_2^2)) X\, dt + B_1 X \circ dW^1 + B_2 X \circ dW^2` with
    commuting matrices, and the exact solution
    :math:`X_t = \exp((A - \frac{1}{2}(B_1^2 + B_2^2)) t + B_1 W^1_t + B_2 W^2_t) X_0`.
    The default matrices are those of the paper.
    """

    ndim = 2
    n_noise = 2

    def __init__(self, brownian_path, T=1.0, n_bridge_terms=0, A=None, B=None, u0=(1.0, 0.0)):
        A = np.array([[1.5, -0.85], [1.275, -0.625]]) if A is None else np.asarray(A)
        B = np.array([[[0.9, -0.2], [0.3, 0.4]], [[2.1, -1.2], [1.8, -0.9]]]) if B is None else np.asarray(B)
        u0 = np.asarray(u0, dtype=float)
        for X in [A, *B]:
            for Y in B:
                if not np.allclose(X @ Y, Y @ X):
                    raise ValueError('the matrices have to commute')
        super().__init__(brownian_path, T, n_bridge_terms)
        self._makeAttributeAndRegister('A', 'B', 'u0', localVars=locals(), readOnly=True)
        self.drift_matrix = A - 0.5 * np.einsum('rij,rjk->ik', B, B)

    def drift(self, u, t):
        return u @ self.drift_matrix.T

    def diffusion(self, u, t):
        return u @ self.B.transpose(0, 2, 1)

    def diffusion_derivative(self, u, t):
        return u @ (self.B[:, None] @ self.B[None]).transpose(0, 1, 3, 2)

    def u_exact(self, t):
        W = self.W(t)
        exponent = self.drift_matrix * t + np.einsum('rs,rij->sij', W, self.B)
        u = mesh(self.init)
        u[:] = expm(exponent) @ self.u0
        return u


class OrnsteinUhlenbeck(SDEProblem):
    r"""
    The Ornstein-Uhlenbeck process :math:`dX = \lambda (\mu - X)\, dt + \sigma \circ dW` (Itô and Stratonovich form
    agree). Its exact solution for the piecewise linear interpolant of the sampled paths,

    .. math::

        X_t = \mu + (X_0 - \mu) e^{-\lambda t} + \sigma \int_0^t e^{-\lambda (t - s)}\, dW_s,

    converges to the solution for the Brownian paths as their grid is refined.
    """

    def __init__(self, brownian_path, T=1.0, n_bridge_terms=0, lam=2.0, mu=1.0, sigma=0.5, u0=0.5):
        super().__init__(brownian_path, T, n_bridge_terms)
        self._makeAttributeAndRegister('lam', 'mu', 'sigma', 'u0', localVars=locals(), readOnly=True)

    def drift(self, u, t):
        return self.lam * (self.mu - u)

    def diffusion(self, u, t):
        return np.full((1, *u.shape), self.sigma)

    def diffusion_derivative(self, u, t):
        return np.zeros((1, 1, *u.shape))

    def u_exact(self, t):
        n, h, lam = self._fine_index(t), self.dt_fine, self.lam
        # integral of exp(-lam (t - s)) over each fine step, divided by its length h
        weights = np.exp(-lam * (t - h * np.arange(1, n + 1))) * -np.expm1(-lam * h) / (lam * h) if n else np.zeros(0)
        u = mesh(self.init)
        u[:, 0] = self.mu + (self.u0 - self.mu) * np.exp(-lam * t)
        u[:, 0] += self.sigma * np.diff(self.brownian_path[0, :, : n + 1], axis=-1) @ weights
        return u

    def solve_sbb_exactly(self, dt, Tend):
        r"""
        Exact solution of the random ODE with the smooth Brownian bridges of this problem on steps of size dt, so
        that its difference to :meth:`u_exact` is the error of the bridges alone.

        On a step, :math:`x(\Delta T) = \mu + (x_0 - \mu) e^{-\lambda \Delta T}
        + \sigma \int_0^{\Delta T} e^{-\lambda (\Delta T - s)} \frac{dB}{ds}\, ds`, where the integral of
        :math:`e^{-\lambda(\Delta T - s)} \cos(j \pi s / \Delta T)` is
        :math:`\lambda ((-1)^j - e^{-\lambda \Delta T}) / (\lambda^2 + (j \pi / \Delta T)^2)`.
        """
        lam, decay = self.lam, np.exp(-self.lam * dt)
        j = np.arange(1, self.n_bridge_terms + 1)
        cosine_integrals = lam * ((-1.0) ** j - decay) / (lam**2 + (j * np.pi / dt) ** 2)
        u = np.asarray(self.u_exact(0.0))[:, 0]
        for i in range(round(Tend / dt)):
            self.set_step(i * dt, dt)
            noise = self.eta0[0] * (1 - decay) / lam + np.sqrt(2.0) * self.xi[0] @ cosine_integrals
            u = self.mu + (u - self.mu) * decay + self.sigma * noise / np.sqrt(dt)
        return u[:, None]
