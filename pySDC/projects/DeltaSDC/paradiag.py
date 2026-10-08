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

The node-local solve needs nothing from this project: the stock heat problem with
``dtype='complex128'`` and the library's ``solve_precision='complex64'`` solves at single precision
while the state stays double, by a sparse solve or, on a periodic grid, by FFT. What is here is the
transform at reduced precision, :class:`controller_ParaDiag_reduced_transform`.

Unlike the PETSc and FEniCS routes in this project, nothing here is emulated: the solves and the
matmul all carry ``complex64`` through, on the CPU and on a GPU alike, so the reduced-precision runs
really are single-precision arithmetic. ParaDiag diagonalises in time, so its working type is
*complex*, and reduced precision therefore means ``complex64`` rather than ``float32``.
"""

import numpy as np

from pySDC.implementations.controller_classes.controller_ParaDiag_nonMPI import controller_ParaDiag_nonMPI

FULL = np.dtype('complex128')
"""Working precision. ParaDiag diagonalises in time, so it is complex even for a real problem."""


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
