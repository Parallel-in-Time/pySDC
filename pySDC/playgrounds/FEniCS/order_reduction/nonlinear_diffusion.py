"""Manufactured nonlinear-diffusion problems for the order-reduction study."""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

from pySDC.core.problem import Problem
from pySDC.implementations.datatype_classes.mesh import mesh


class manufactured_nonlinear_diffusion(Problem):
    r"""One-dimensional nonlinear diffusion with manufactured forcing.

    The equation is

    .. math::
        u_t = \partial_x(a(u)u_x) + f(x,t),
        \qquad a(u)=1+\gamma u^2.

    The exact solution is ``u=cos(t)*s(x)+c``.  The forcing is manufactured
    using the same finite-volume flux operator used by the time integrator,
    isolating the temporal convergence behavior.
    """

    dtype_u = mesh
    dtype_f = mesh

    def __init__(self, nvars=127, gamma=1.0, c=0.25, profile='sine', newton_maxiter=50, newton_tol=1e-12):
        if (nvars + 1) % 2:
            raise ValueError('nvars must be 2^p - 1 for the study setup')
        if profile not in ('sine', 'cosine'):
            raise ValueError(f'unknown profile {profile!r}')
        super().__init__((nvars, None, np.dtype('float64')))
        self._makeAttributeAndRegister(
            'nvars', 'gamma', 'c', 'profile', 'newton_maxiter', 'newton_tol', localVars=locals(), readOnly=True
        )
        self.dx = 1.0 / (nvars + 1)
        self.x = np.arange(1, nvars + 1, dtype=float) * self.dx

    def _shape(self, x):
        return np.sin(np.pi * x) if self.profile == 'sine' else np.cos(np.pi * x)

    def _boundary(self, t):
        return np.array([self.c + np.cos(t) * self._shape(0.0), self.c + np.cos(t) * self._shape(1.0)])

    def _flux(self, values):
        midpoint = 0.5 * (values[:-1] + values[1:])
        diffusivity = 1.0 + self.gamma * midpoint**2
        return diffusivity * (values[1:] - values[:-1]) / self.dx

    def _diffusion(self, values):
        return (self._flux(values)[1:] - self._flux(values)[:-1]) / self.dx

    def _diffusion_jacobian(self, values):
        """Jacobian of the interior flux divergence with respect to interiors."""
        n = self.nvars
        dleft = np.zeros(n + 1)
        dright = np.zeros(n + 1)
        for face in range(n + 1):
            left, right = values[face], values[face + 1]
            midpoint = 0.5 * (left + right)
            diffusivity = 1.0 + self.gamma * midpoint**2
            dleft[face] = self.gamma * midpoint * (right - left) / self.dx - diffusivity / self.dx
            dright[face] = self.gamma * midpoint * (right - left) / self.dx + diffusivity / self.dx
        diagonal = (dleft[1:] - dright[:-1]) / self.dx
        upper = dright[1:-1] / self.dx
        lower = -dleft[1:-1] / self.dx
        return sp.diags([lower, diagonal, upper], [-1, 0, 1], format='csc')

    def _forcing(self, t):
        exact = np.empty(self.nvars + 2)
        exact[1:-1] = np.cos(t) * self._shape(self.x) + self.c
        exact[[0, -1]] = self._boundary(t)
        ut = -np.sin(t) * self._shape(self.x)
        return ut - self._diffusion(exact)

    def eval_f(self, u, t):
        values = np.empty(self.nvars + 2)
        values[1:-1] = u
        values[[0, -1]] = self._boundary(t)
        result = self.dtype_f(self.init, val=0.0)
        result[:] = self._diffusion(values) + self._forcing(t)
        return result

    def solve_system(self, rhs, factor, u0, t):
        u = self.dtype_u(u0)
        boundary = self._boundary(t)
        identity = sp.eye(self.nvars, format='csc')
        for _ in range(self.newton_maxiter):
            values = np.empty(self.nvars + 2)
            values[1:-1] = u
            values[[0, -1]] = boundary
            residual = u - rhs - factor * (self._diffusion(values) + self._forcing(t))
            if np.linalg.norm(residual, np.inf) < self.newton_tol:
                break
            jacobian = identity - factor * self._diffusion_jacobian(values)
            u -= spsolve(jacobian, residual)
        else:
            raise RuntimeError('Newton iteration did not converge')
        return self.dtype_u(u)

    def u_exact(self, t):
        result = self.dtype_u(self.init, val=0.0)
        result[:] = np.cos(t) * self._shape(self.x) + self.c
        return result


class manufactured_nonlinear_diffusion_sine(manufactured_nonlinear_diffusion):
    """Nonlinear diffusion with time-independent boundary values."""

    def __init__(self, **kwargs):
        super().__init__(profile='sine', **kwargs)


class manufactured_nonlinear_diffusion_cosine(manufactured_nonlinear_diffusion):
    """Nonlinear diffusion with time-dependent boundary values."""

    def __init__(self, **kwargs):
        super().__init__(profile='cosine', **kwargs)


class manufactured_nonlinear_diffusion_cosine_lift(manufactured_nonlinear_diffusion_cosine):
    r"""Cosine nonlinear diffusion solved for the homogeneous lifted state."""

    def _lift(self, t):
        return self.c + (1.0 - 2.0 * self.x) * np.cos(t)

    def _lift_t(self, t):
        return -(1.0 - 2.0 * self.x) * np.sin(t)

    def eval_f(self, v, t):
        physical = v + self._lift(t)
        values = np.empty(self.nvars + 2)
        values[1:-1] = physical
        values[[0, -1]] = self._boundary(t)
        result = self.dtype_f(self.init, val=0.0)
        result[:] = self._diffusion(values) + self._forcing(t) - self._lift_t(t)
        return result

    def solve_system(self, rhs, factor, u0, t):
        v = self.dtype_u(u0)
        lift = self._lift(t)
        identity = sp.eye(self.nvars, format='csc')
        for _ in range(self.newton_maxiter):
            physical = v + lift
            values = np.empty(self.nvars + 2)
            values[1:-1] = physical
            values[[0, -1]] = self._boundary(t)
            residual = v - rhs - factor * (self._diffusion(values) + self._forcing(t) - self._lift_t(t))
            if np.linalg.norm(residual, np.inf) < self.newton_tol:
                break
            jacobian = identity - factor * self._diffusion_jacobian(values)
            v -= spsolve(jacobian, residual)
        else:
            raise RuntimeError('Newton iteration did not converge')
        return self.dtype_u(v)

    def lift(self, t):
        result = self.dtype_u(self.init, val=0.0)
        result[:] = self._lift(t)
        return result

    def u_exact_lifted(self, t):
        result = self.dtype_u(self.init, val=0.0)
        result[:] = np.cos(t) * (np.cos(np.pi * self.x) - 1.0 + 2.0 * self.x)
        return result

    def u_exact(self, t):
        return super().u_exact(t)
