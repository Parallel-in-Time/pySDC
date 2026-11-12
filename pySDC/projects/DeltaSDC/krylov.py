"""
Conjugate gradients in fp64, fp32 or fp16, on the CPU with NumPy or on a GPU with CuPy.

The vectors and the application of the operator are in the chosen format. Inner products accumulate in fp32
(fp64 for fp64), as GPU libraries do for fp16: the products of two fp16 numbers are exact in fp32. The operator
is given as a function, so that it can be a stencil without an assembled matrix.
"""

import numpy as np

from pySDC.implementations.problem_classes.HeatEquation_ND_FD import heatNd_unforced

DTYPES = {'fp64': 'float64', 'fp32': 'float32', 'fp16': 'float16'}


_GPU_KERNELS = {}
# no fused multiply-adds: every product and sum is rounded on its own, as in the unfused code
NO_FMA = ('--fmad=false',)


def _gpu_kernels():
    """Fused kernels for CG on a GPU: the inner product with a wider accumulator, and the two vector updates."""
    if not _GPU_KERNELS:
        import cupy as cp

        # the products, widened to the accumulator, in one kernel; summed by cp.sum, which uses CUB as the unfused
        # code does, so the inner product is the same number. (A ReductionKernel would reduce a whole vector to a
        # scalar with a single thread block, which is far slower.)
        for name, accumulate, c_type in (('dot32', 'float32', 'float'), ('dot64', 'float64', 'double')):
            product = cp.ElementwiseKernel(
                'T u, T v',
                f'{accumulate} z',
                f'z = {c_type}(u) * {c_type}(v)',
                f'pysdc_cg_product_{name}',
                options=NO_FMA,
            )
            _GPU_KERNELS[name] = lambda u, v, product=product: cp.sum(product(u, v))
        # x += alpha p and r -= alpha Ap, each operation rounded to the format, as the unfused code does
        _GPU_KERNELS['update_xr'] = cp.ElementwiseKernel(
            'T p, T Ap, T alpha',
            'T x, T r',
            'x = x + T(alpha * p); r = r - T(alpha * Ap)',
            'pysdc_cg_update_xr',
            options=NO_FMA,
        )
        _GPU_KERNELS['update_p'] = cp.ElementwiseKernel(
            'T r, T beta', 'T p', 'p = r + T(beta * p)', 'pysdc_cg_update_p', options=NO_FMA
        )
    return _GPU_KERNELS


def cg(apply, b, fmt, rtol, maxiter, xp=np, fused=False):
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
    fused : bool
        On a GPU, use fused kernels for the inner products and the vector updates: the same operations with the
        same roundings and the same summation, in fewer passes over memory.

    Returns
    -------
    tuple
        The solution (in the format), the number of iterations and the relative residual reached.
    """
    dtype = xp.dtype(DTYPES[fmt])
    accumulate = xp.float64 if fmt == 'fp64' else xp.float32

    fused = fused and xp is not np
    if fused:
        kernels = _gpu_kernels()
        reduce = kernels['dot64' if fmt == 'fp64' else 'dot32']

        def dot(u, v):
            return float(reduce(u, v))

    else:

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
        if fused:
            kernels['update_xr'](p, Ap, alpha, x, r)
        else:
            x += alpha * p
            r -= alpha * Ap
        rr, rr_old = dot(r, r), rr
        if not 0 < rr < np.inf:  # the residual has vanished or overflowed in the format
            break
        if fused:
            kernels['update_p'](r, dtype.type(rr / rr_old), p)
        else:
            p = r + dtype.type(rr / rr_old) * p
        k += 1
    return x, k, float(np.sqrt(rr / norm_b))


_STENCIL = {}


def _gpu_stencil(cp):
    """The Jacobi-scaled 1D stencil x - c (x[k-1] + x[k+1]), fused, with the summation order of the roll version."""
    if not _STENCIL:
        body = """
        int n = x.size(), k = i;
        T s = x[k == 0 ? n - 1 : k - 1] + x[k == n - 1 ? 0 : k + 1];
        y = x[k] - T(c * s)
        """
        kernel = cp.ElementwiseKernel('raw T x, T c', 'T y', body, 'pysdc_heat_cg_stencil', options=('--fmad=false',))
        _STENCIL['1d'] = lambda x, c: kernel(x, c, cp.empty_like(x))
    return _STENCIL['1d']


class heat_cg(heatNd_unforced):
    """
    The 1D periodic heat equation with second-order differences, whose solves are CG on the stencil.

    The system is divided by its constant diagonal first (Jacobi scaling), which keeps fp16 in range, and the
    right-hand side is scaled to a maximum of one. CG runs in ``solve_format`` to the relative residual
    ``cg_rtol``; the iterations of every solve are appended to ``cg_iterations``.
    """

    def __init__(self, solve_format='fp64', cg_rtol=1e-12, cg_maxiter=1000, **kwargs):
        super().__init__(**kwargs)
        if len(self.nvars) != 1 or self.bc != 'periodic' or self.order != 2:
            raise ValueError('heat_cg needs a 1D periodic grid and second-order differences')
        self._makeAttributeAndRegister('solve_format', 'cg_rtol', 'cg_maxiter', localVars=locals())
        self.cg_iterations = []

    def solve_system(self, rhs, factor, u0, t):
        xp, n = self.xp, self.nvars[0]
        a = factor * self.nu * n**2
        diagonal = 1 + 2 * a
        coupling = xp.dtype(DTYPES[self.solve_format]).type(a / diagonal)
        b = rhs.view(xp.ndarray).reshape(-1) / diagonal
        peak = float(xp.abs(b).max())
        me = self.dtype_u(self.init)
        if peak == 0:
            me[:] = 0.0
            return me
        gpu = xp is not np  # fused kernels on a GPU, bitwise the same as the array operations
        apply = (
            (lambda x: _gpu_stencil(xp)(x, coupling))
            if gpu
            else (lambda x: x - coupling * (xp.roll(x, 1) + xp.roll(x, -1)))
        )
        x, iterations, _ = cg(apply, b / peak, self.solve_format, self.cg_rtol, self.cg_maxiter, xp=xp, fused=gpu)
        self.cg_iterations.append(iterations)
        me[:] = (x.astype(xp.float64) * peak).reshape(me.shape)
        return me
