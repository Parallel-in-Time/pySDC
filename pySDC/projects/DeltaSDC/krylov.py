"""
Conjugate gradients in fp64, fp32 or fp16, on the CPU with NumPy or on a GPU with CuPy.

The vectors and the application of the operator are in the chosen format. Inner products accumulate in fp32
(fp64 for fp64), as GPU libraries do for fp16: the products of two fp16 numbers are exact in fp32. The operator
is given as a function, so that it can be a stencil without an assembled matrix.
"""

import numpy as np

DTYPES = {'fp64': 'float64', 'fp32': 'float32', 'fp16': 'float16'}


def cg(apply, b, fmt, rtol, maxiter, xp=np):
    """
    Solve A x = b for a symmetric positive definite A, starting from zero.

    Parameters
    ----------
    apply : callable
        Computes A x in the format of x.
    b : array
        The right-hand side, of order one (fp16 stops at 65504).
    fmt : str
        'fp64', 'fp32' or 'fp16'.
    rtol : float
        Stop once the recursively updated residual is below rtol times the norm of b.
    maxiter : int
        Stop after this many iterations at the latest.
    xp : module
        numpy or cupy.

    Returns
    -------
    tuple
        The solution (in the format), the number of iterations and the relative residual reached.
    """
    dtype = xp.dtype(DTYPES[fmt])
    accumulate = xp.float64 if fmt == 'fp64' else xp.float32

    def dot(u, v):
        return float(xp.sum(u.astype(accumulate) * v.astype(accumulate)))

    x = xp.zeros_like(b, dtype=dtype)
    r = b.astype(dtype)
    p = r.copy()
    rr = norm_b = dot(r, r)
    k = 0
    while k < maxiter and rr > rtol**2 * norm_b:
        Ap = apply(p)
        pAp = dot(p, Ap)
        if pAp <= 0:  # the search direction has underflowed in the format: no further progress possible
            break
        alpha = dtype.type(rr / pAp)
        x += alpha * p
        r -= alpha * Ap
        rr, rr_old = dot(r, r), rr
        if not 0 < rr < np.inf:  # the residual has vanished or overflowed in the format
            break
        p = r + dtype.type(rr / rr_old) * p
        k += 1
    return x, k, float(np.sqrt(rr / norm_b))
