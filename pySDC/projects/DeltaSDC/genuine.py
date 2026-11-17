"""
Node-local solves genuinely in fp32 or fp16 arithmetic, on any CPU.

A dense LU with partial pivoting in which every operation is rounded to the format: computed in fp32,
then correctly rounded, which for +, -, * and / is exactly the format's own arithmetic, since fp32
carries more than twice the bits of fp16 plus two (there is no fused multiply-add). Matrix
and right-hand side are rounded on input. The right-hand side arrives at unit size (the
correction-solve contract); the matrix is not scaled, so it must fit the format's range.

This is a backward-stable solver in the format, and slow: meant for 1D test problems, not for production.
"""

import numpy as np


def rounder(fmt):
    if fmt == 'fp32':
        return lambda x: np.asarray(x, dtype=np.float32)
    if fmt == 'fp16':
        return lambda x: np.asarray(x, dtype=np.float32).astype(np.float16).astype(np.float32)
    raise ValueError(fmt)


def lu(B, fmt):
    r = rounder(fmt)
    A = r(B).copy()
    n = len(A)
    piv = np.arange(n)
    for k in range(n - 1):
        p = k + int(np.argmax(np.abs(A[k:, k])))
        if p != k:
            A[[k, p]] = A[[p, k]]
            piv[[k, p]] = piv[[p, k]]
        A[k + 1 :, k] = r(A[k + 1 :, k] / A[k, k])
        A[k + 1 :, k + 1 :] = r(A[k + 1 :, k + 1 :] - r(np.outer(A[k + 1 :, k], A[k, k + 1 :])))
    if not np.all(np.isfinite(A)):
        raise FloatingPointError(f'{fmt} LU overflowed')
    return A, piv


def lu_solve(factors, b, fmt):
    A, piv = factors
    r = rounder(fmt)
    x = r(np.asarray(b)[piv]).copy()
    n = len(x)
    for k in range(n - 1):  # L, unit diagonal, column by column
        x[k + 1 :] = r(x[k + 1 :] - r(A[k + 1 :, k] * x[k]))
    for k in range(n - 1, -1, -1):
        x[k] = r(x[k] / A[k, k])
        x[:k] = r(x[:k] - r(A[:k, k] * x[k]))
    return x.astype(np.float64)


def genuine(problem_class):
    """
    Subclass ``problem_class`` (with a matrix ``A`` and a linear ``solve_system``) so that its solves run genuinely
    in ``solve_format``: 'fp64' (the parent's solve), 'fp32' or 'fp16' (the LU above). Each row of the matrix is
    divided by its diagonal entry and the right-hand side by its largest entry first, as a practical fp16 LU would,
    to stay in fp16's range.
    """

    class lowprec(problem_class):
        def __init__(self, solve_format='fp64', **kwargs):
            super().__init__(**kwargs)
            self._makeAttributeAndRegister('solve_format', localVars=locals())
            self._factors = {}

        def solve_system(self, rhs, factor, u0, t):
            fmt = self.solve_format
            if fmt == 'fp64':
                return super().solve_system(rhs, factor, u0, t)
            if factor not in self._factors:
                B = np.eye(self.A.shape[0]) - factor * self.A.toarray()
                scale = 1 / np.diag(B)
                self._factors[factor] = (lu(scale[:, None] * B, fmt), scale)
            factors, scale = self._factors[factor]
            b = np.asarray(rhs).reshape(-1) * scale
            peak = float(np.abs(b).max())
            me = self.dtype_u(self.init)
            me[:] = 0.0 if peak == 0 else (lu_solve(factors, b / peak, fmt) * peak).reshape(me.shape)
            return me

    lowprec.__name__ = lowprec.__qualname__ = f'{problem_class.__name__}_genuine'
    return lowprec
