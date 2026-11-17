r"""
The 3D periodic heat equation with second-order differences, whose node solves are geometric multigrid in a chosen
format, on a CPU or a GPU.

The right-hand side :math:`f(u) = \nu \Delta_h u` is the 7-point stencil, evaluated without a matrix in fp64 by the
stencil kernel of the solver. A node solve :math:`(I - s \Delta_h) x = b`, :math:`s = \text{factor}\,\nu`, is
:class:`pySDC.helpers.multigrid.GeometricMultigrid` in ``solve_format`` from a zero initial guess to the relative
residual ``mg_rtol``, and it returns its result in that format, as the correction-solve contract of the delta-form
sweepers allows.
"""

import numpy as np

from pySDC.core.problem import Problem, WorkCounter
from pySDC.helpers.multigrid import GeometricMultigrid
from pySDC.implementations.datatype_classes.mesh import mesh

DTYPES = {'fp64': np.float64, 'fp32': np.float32, 'fp16': np.float16}


class heat3d_mg(Problem):
    r"""
    :math:`u_t = \nu \Delta u` on :math:`[0, 1)^3`, periodic, with second-order differences on :math:`N^3` points,
    and node solves by :class:`GeometricMultigrid` in ``solve_format``.

    ``mg_iterations`` collects the V-cycles of every solve, and with ``time_solves=True`` on a GPU, ``solve_events``
    the CUDA events around every solve (for the share of the run time the solves take, without synchronizing).
    """

    dtype_u = mesh
    dtype_f = mesh

    def __init__(
        self, N=64, nu=1.0, solve_format='fp64', mg_rtol=5e-3, mg_maxiter=50, useGPU=False, time_solves=False, fma=False
    ):
        if useGPU:
            import cupy as cp
            from pySDC.implementations.datatype_classes.cupy_mesh import cupy_mesh

            self.xp, self.dtype_u, self.dtype_f = cp, cupy_mesh, cupy_mesh
        else:
            self.xp = np
        super().__init__(init=((N, N, N), None, np.dtype('float64')))
        self._makeAttributeAndRegister(
            'N',
            'nu',
            'solve_format',
            'mg_rtol',
            'mg_maxiter',
            'useGPU',
            'time_solves',
            'fma',
            localVars=locals(),
            readOnly=True,
        )
        self.multigrid = GeometricMultigrid(N, 3, xp=self.xp, fma=fma)
        self.mg_iterations, self.solve_events = [], []
        self.work_counters['rhs'] = WorkCounter()

    def eval_f(self, u, t):
        f = self.dtype_f(self.init)
        self.multigrid.laplacian(u.view(self.xp.ndarray), self.nu, out=f.view(self.xp.ndarray))
        self.work_counters['rhs']()
        return f

    def eval_f_increment(self, base, delta, t):
        """f(base + delta) - f(base) = nu Lap delta, for a linear operator."""
        f = self.dtype_f(self.init)
        self.multigrid.laplacian(delta.view(self.xp.ndarray), self.nu, out=f.view(self.xp.ndarray))
        return f

    def u_exact(self, t):
        """A Gaussian bump of width 0.1 at the centre, at t = 0 only."""
        assert t == 0, 'only the initial value is known'
        me = self.dtype_u(self.init)
        g = np.arange(self.N) / self.N
        bump = np.exp(-(((g - 0.5) / 0.1) ** 2))
        me[:] = self.xp.asarray(bump[:, None, None] * bump[None, :, None] * bump[None, None, :])
        return me

    def solve_system(self, rhs, factor, u0, t):
        """
        (I - factor nu Lap) x = rhs by multigrid in solve_format, from a zero guess, to mg_rtol.

        The result is in solve_format, as a plain array unless that is fp64: this is the correction solve of the
        delta-form sweepers, whose contract allows it, and which widen it once.
        """
        xp = self.xp
        if self.time_solves and self.multigrid.ops.gpu:
            start, end = xp.cuda.Event(), xp.cuda.Event()
            start.record()
        b = rhs.view(xp.ndarray).astype(DTYPES[self.solve_format])
        x, cycles = self.multigrid.solve(b, factor * self.nu, self.mg_rtol, self.mg_maxiter)
        self.mg_iterations.append(cycles)
        if self.solve_format == 'fp64':
            me = self.dtype_u(self.init)
            me[:] = x
        else:
            me = x  # in the solve format, as the correction-solve contract allows: the sweeper widens it once
        if self.time_solves and self.multigrid.ops.gpu:
            end.record()
            self.solve_events.append((start, end))
        return me
