r"""
ParaDiag at reduced precision.

ParaDiag needs no delta reformulation: it is already in the form this project exists to construct.
Its iteration is

.. math::
    r^k = b - \mathcal{C}u^k, \qquad
    \delta^k = \mathcal{C}_\alpha^{-1} r^k, \qquad
    u^{k+1} = u^k + \delta^k,

with :math:`\mathcal{C}` the composite collocation operator and :math:`\mathcal{C}_\alpha` its
alpha-circulant approximation, applied by diagonalising across the steps. That is iterative
refinement at the level of the whole block, so the two properties the delta-form sweepers had to be
rewritten for hold here by construction:

1. the quantity handed to the node-local solver is the **increment**, whose magnitude falls with
   the iteration, and
2. the residual that produces it is formed once, in working precision, outside the solve.

The consequence is that *everything inside the preconditioner* may run at reduced precision without
capping the attainable accuracy -- not only the node-local solve, but the weighted FFT across the
steps as well. Only the residual, the stored solution and the update :math:`u + \delta` have to stay
at working precision.

The weighted transform is worth naming separately, because the obvious guess is that it cannot take
reduced precision. It is deliberately ill-conditioned: ``get_J_inv_matrix`` weights entry
:math:`l` by :math:`\alpha^{l/L}`, so the inverse transform amplifies by up to
:math:`\alpha^{-(L-1)/L} \approx 1/\alpha`, and one expects a floor near
:math:`\varepsilon/\alpha`. That does not happen, because the amplification acts on the increment,
which is itself shrinking. See ``tests/test_paradiag.py``, which pins both halves of this and
carries the control that shows the measurement can fail.

Unlike the PETSc and FEniCS routes in this project, nothing here is emulated: the FFT (periodic),
the sparse solver (otherwise) and the matmul all carry ``complex64`` through, on the CPU and on a
GPU alike, so the reduced-precision runs really are single-precision arithmetic. ParaDiag diagonalises in time, so its working type is *complex*, and
reduced precision therefore means ``complex64`` rather than ``float32``.
"""

import numpy as np
import scipy.fft

from pySDC.implementations.controller_classes.controller_ParaDiag_nonMPI import controller_ParaDiag_nonMPI
from pySDC.core.problem import WorkCounter
from pySDC.implementations.problem_classes.HeatEquation_ND_FD import heatNd_forced

FULL = np.dtype('complex128')
"""Working precision. ParaDiag diagonalises in time, so it is complex even for a real problem."""


class heat_paradiag(heatNd_forced):
    """
    ``heatNd_forced`` prepared for ParaDiag, with the node-local solve at a chosen precision.

    Two things separate this from the stock problem. The state is complex, because diagonalising
    across the steps makes it so, and ``solve_jacobian`` solves
    :math:`(I - \\text{factor}\\,A)\\,x = r` at ``solve_precision`` instead of inheriting the
    double-precision solve. The factor is complex here -- it is an eigenvalue of the circulant
    times :math:`\\Delta t` -- so a sparse operator has to be assembled per call. With periodic
    boundaries none is: the FFT diagonalises :math:`A`, and the solve is a division in Fourier
    space. That path is what runs on a GPU; ``spsolve`` there is too slow to be worth timing.

    Parameters
    ----------
    solve_precision : dtype-like or None, optional
        Working precision of the node-local solve. ``None`` keeps ``complex128``.
    """

    def __init__(self, solve_precision=None, **kwargs):
        super().__init__(**kwargs)
        self._makeAttributeAndRegister('solve_precision', localVars=locals())

        # ParaDiag diagonalises across the steps, so the state is complex whatever the problem is
        self.init = tuple([*self.init[:2]] + [FULL])

        self._solve_dtype = FULL if solve_precision is None else np.dtype(solve_precision)
        self.work_counters['paradiag_solve'] = WorkCounter()

        # Complex once, here: the state is complex, and a real sparse matrix applied to a complex
        # vector is converted on every product by CuPy, which would dominate the residual.
        self.A = self.A.astype(FULL)

        # A periodic finite-difference operator is circulant, so the FFT diagonalises it exactly and
        # its first column transforms to its eigenvalues. The solve is then two FFTs and a division,
        # which is also the one place where single precision is several times faster on a GPU.
        # scipy.fft rather than numpy.fft on the CPU: NumPy 1 computes a complex64 FFT in double.
        self._fft = scipy.fft if self.xp is np else self.xp.fft
        if self.bc == 'periodic':
            first_column = self.xp.zeros(self.A.shape[0], dtype=FULL)
            first_column[0] = 1.0
            eigenvalues = self._fft.fftn((self.A @ first_column).reshape(self.nvars))
            self._eigenvalues = eigenvalues.astype(self._solve_dtype)

    def solve_jacobian(self, rhs, factor, u=None, u0=None, t=0, **kwargs):
        r"""
        Solve :math:`(I - \text{factor}\,A)\,x = \text{rhs}` at the configured precision.

        In ParaDiag ``rhs`` is a residual and ``x`` an increment, so an error of relative size
        :math:`\varepsilon` here is an absolute error :math:`\varepsilon|\delta|`, which vanishes
        with the iteration. That is what makes reducing this precision safe, and it is a property
        of ParaDiag's formulation rather than of anything done here.

        Parameters
        ----------
        rhs : dtype_u
            Right-hand side, a residual in Fourier space across the steps.
        factor : complex
            Circulant eigenvalue times the step size.
        u, u0, t
            Accepted for interface compatibility; a linear problem needs none of them.

        Returns
        -------
        dtype_u
            The increment, cast back to working precision.
        """
        dtype = self._solve_dtype
        rhs = rhs.view(self.xp.ndarray).astype(dtype)
        if self.bc == 'periodic':
            solution = self._fft.ifftn(self._fft.fftn(rhs) / (1 - dtype.type(factor) * self._eigenvalues))
        else:
            identity = self.xsp.eye(self.A.shape[0], dtype=dtype, format='csr')
            operator = (identity - dtype.type(factor) * self.A.astype(dtype)).tocsr()
            solution = self.linalg.spsolve(operator, rhs.flatten())
        self.work_counters['paradiag_solve']()

        me = self.dtype_u(self.init)
        me[:] = solution.astype(FULL).reshape(self.nvars)
        return me


class controller_ParaDiag_reduced_transform(controller_ParaDiag_nonMPI):
    """
    ParaDiag whose weighted FFT and iFFT across the steps run at a chosen precision.

    Set ``transform_precision`` in the controller parameters. The default keeps ``complex128``, so
    this is the stock controller up to round-off unless asked otherwise: the same transform as one
    matmul instead of a loop of vector updates. Runs wherever the problem's ``xp`` does.

    Both quantities this transforms -- the residual and the increment -- are small and shrinking, so
    reducing the precision of the transform costs convergence rate rather than attainable accuracy,
    despite the transform's :math:`1/\\alpha` amplification.
    """

    def apply_matrix(self, mat, quantity):
        """
        Apply a square L x L matrix across the steps, in place, at ``transform_precision``.

        Args:
            mat: square matrix with as many rows as there are steps
            quantity (str): 'residual' or 'increment', the level attribute to transform
        """
        dtype = np.dtype(getattr(self.params, 'transform_precision', FULL))
        xp = self.MS[0].levels[0].prob.xp

        L = len(self.MS)
        assert np.allclose(mat.shape, L)

        fields = [getattr(S.levels[0], quantity) for S in self.MS]
        M = len(fields[0])

        # One (L, M * ndof) block, so the transform is a single matmul, at any precision -- including
        # full, so that the precisions compared differ in nothing but the precision. `view` strips the
        # datatype wrapper, which CuPy will not stack.
        block = xp.stack([xp.stack([field[m].view(xp.ndarray).reshape(-1) for m in range(M)]) for field in fields])
        result = (xp.asarray(mat, dtype=dtype) @ block.astype(dtype).reshape(L, -1)).reshape(block.shape)

        for i, field in enumerate(fields):
            for m in range(M):
                field[m][:] = result[i, m].astype(FULL).reshape(field[m].shape)
