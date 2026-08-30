"""Manufactured semilinear reaction-diffusion problems for order reduction."""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

from pySDC.core.problem import Problem
from pySDC.implementations.datatype_classes.mesh import mesh


class manufactured_reaction_diffusion(Problem):
    """One-dimensional manufactured semilinear reaction-diffusion problem.

    The physical equation is

    .. math::
        u_t = \\nu u_{xx} + \\lambda u(1-u) + f(x,t),

    with manufactured solution ``u = cos(t) s(x) + c``.  ``sine`` uses
    ``s(x)=sin(pi*x)`` and consequently has time-independent boundary values;
    ``cosine`` uses ``s(x)=cos(pi*x)`` and has time-dependent boundary values.
    """

    dtype_u = mesh
    dtype_f = mesh

    def __init__(
        self,
        nvars=127,
        nu=0.1,
        reaction=1.0,
        c=0.25,
        profile='sine',
        newton_maxiter=50,
        newton_tol=1e-12,
    ):
        if (nvars + 1) % 2:
            raise ValueError('nvars must be 2^p - 1 for the study setup')
        if profile not in ('sine', 'cosine'):
            raise ValueError(f'unknown profile {profile!r}')

        super().__init__((nvars, None, np.dtype('float64')))
        self._makeAttributeAndRegister(
            'nvars', 'nu', 'reaction', 'c', 'profile', 'newton_maxiter', 'newton_tol', localVars=locals(), readOnly=True
        )
        self.dx = 1.0 / (nvars + 1)
        self.x = np.arange(1, nvars + 1, dtype=float) * self.dx
        self.x_all = np.arange(nvars + 2, dtype=float) * self.dx
        self._laplace = sp.diags(
            [np.ones(nvars - 1), -2.0 * np.ones(nvars), np.ones(nvars - 1)], [-1, 0, 1], format='csc'
        ) / self.dx**2

    def _shape(self, x):
        return np.sin(np.pi * x) if self.profile == 'sine' else np.cos(np.pi * x)

    def _shape_xx(self, x):
        return -np.pi**2 * self._shape(x)

    def _boundary(self, t):
        return np.array([self.c + np.cos(t) * self._shape(0.0), self.c + np.cos(t) * self._shape(1.0)])

    def _forcing(self, t):
        q = np.cos(t)
        u = q * self._shape(self.x) + self.c
        ut = -np.sin(t) * self._shape(self.x)
        values = np.empty(self.nvars + 2)
        values[1:-1] = u
        values[[0, -1]] = self._boundary(t)
        # Manufacture against the discrete operator. This removes the spatial
        # truncation error from the temporal order study.
        uxx = self._laplace_with_boundary(values)
        return ut - self.nu * uxx - self.reaction * u * (1.0 - u)

    def _laplace_with_boundary(self, values):
        """Apply the centered Laplacian to interior values including BCs."""
        return (values[:-2] - 2.0 * values[1:-1] + values[2:]) / self.dx**2

    def eval_f(self, u, t):
        """Evaluate diffusion, reaction, and manufactured forcing."""
        values = np.empty(self.nvars + 2)
        values[1:-1] = u
        values[[0, -1]] = self._boundary(t)
        f = self.dtype_f(self.init, val=0.0)
        f[:] = self.nu * self._laplace_with_boundary(values) + self.reaction * u * (1.0 - u) + self._forcing(t)
        return f

    def solve_system(self, rhs, factor, u0, t):
        """Newton solve for the implicit reaction-diffusion node equation."""
        u = self.dtype_u(u0)
        boundary = self._boundary(t)
        identity = sp.eye(self.nvars, format='csc')
        for _ in range(self.newton_maxiter):
            values = np.empty(self.nvars + 2)
            values[1:-1] = u
            values[[0, -1]] = boundary
            reaction = self.reaction * u * (1.0 - u)
            residual = u - rhs - factor * (
                self.nu * self._laplace_with_boundary(values) + reaction + self._forcing(t)
            )
            if np.linalg.norm(residual, np.inf) < self.newton_tol:
                break
            jacobian = identity - factor * (
                self.nu * self._laplace + sp.diags(self.reaction * (1.0 - 2.0 * u), format='csc')
            )
            u -= spsolve(jacobian, residual)
        else:
            raise RuntimeError('Newton iteration did not converge')
        return self.dtype_u(u)

    def u_exact(self, t):
        """Return the physical manufactured solution at time ``t``."""
        result = self.dtype_u(self.init, val=0.0)
        result[:] = np.cos(t) * self._shape(self.x) + self.c
        return result


class manufactured_reaction_diffusion_sine(manufactured_reaction_diffusion):
    """Semilinear problem with time-independent Dirichlet data."""

    def __init__(self, **kwargs):
        super().__init__(profile='sine', **kwargs)


class manufactured_reaction_diffusion_cosine(manufactured_reaction_diffusion):
    """Semilinear problem with time-dependent Dirichlet data."""

    def __init__(self, **kwargs):
        super().__init__(profile='cosine', **kwargs)


class manufactured_reaction_diffusion_cosine_lift(manufactured_reaction_diffusion_cosine):
    r"""Cosine problem solved in the homogeneous lifted variable.

    The physical exact solution remains ``cos(pi*x)*cos(t)+c``.  The controller
    state is ``v = u - E`` with ``E=(1-2*x)*cos(t)+c``.  The transformed reaction
    is evaluated at ``v+E`` and the forcing contains ``-E_t``.
    """

    def _lift(self, t):
        return self.c + (1.0 - 2.0 * self.x) * np.cos(t)

    def _lift_t(self, t):
        return -(1.0 - 2.0 * self.x) * np.sin(t)

    def eval_f(self, v, t):
        physical = v + self._lift(t)
        values = np.empty(self.nvars + 2)
        values[1:-1] = v
        values[[0, -1]] = 0.0
        f = self.dtype_f(self.init, val=0.0)
        f[:] = (
            self.nu * self._laplace_with_boundary(values)
            + self.reaction * physical * (1.0 - physical)
            + self._forcing(t)
            - self._lift_t(t)
        )
        return f

    def solve_system(self, rhs, factor, u0, t):
        """Newton solve for ``v`` with homogeneous boundary values."""
        v = self.dtype_u(u0)
        identity = sp.eye(self.nvars, format='csc')
        lift = self._lift(t)
        for _ in range(self.newton_maxiter):
            physical = v + lift
            values = np.empty(self.nvars + 2)
            values[1:-1] = v
            values[[0, -1]] = 0.0
            residual = v - rhs - factor * (
                self.nu * self._laplace_with_boundary(values)
                + self.reaction * physical * (1.0 - physical)
                + self._forcing(t)
                - self._lift_t(t)
            )
            if np.linalg.norm(residual, np.inf) < self.newton_tol:
                break
            jacobian = identity - factor * (
                self.nu * self._laplace + sp.diags(self.reaction * (1.0 - 2.0 * physical), format='csc')
            )
            v -= spsolve(jacobian, residual)
        else:
            raise RuntimeError('Newton iteration did not converge')
        return self.dtype_u(v)

    def lift(self, t):
        """Return the interior lift used to reconstruct the physical state."""
        result = self.dtype_u(self.init, val=0.0)
        result[:] = self._lift(t)
        return result

    def u_exact_lifted(self, t):
        """Return the exact controller state ``v_exact = u_exact - E``."""
        result = self.dtype_u(self.init, val=0.0)
        result[:] = np.cos(t) * (np.cos(np.pi * self.x) - 1.0 + 2.0 * self.x)
        return result

    def u_exact(self, t):
        """Return the unchanged physical cosine solution."""
        return super().u_exact(t)
