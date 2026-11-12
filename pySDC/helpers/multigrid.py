r"""
Geometric multigrid for :math:`(I - s A_h)\, x = b` with :math:`A_h = \nu \Delta_h - c \cdot \nabla_h`, on a CPU
(NumPy) or a GPU (CuPy).

This is the system an implicit Euler step, and so every node solve of SDC, of an advection-diffusion term
:math:`\nu \Delta u - c \cdot \nabla u` leads to, with :math:`s` the factor of the step. :math:`\Delta_h` is the
second-order Laplacian (3, 5 or 7 points) and :math:`\nabla_h` the first-order upwind gradient, each component of
:math:`c` differenced on the side it comes from, on :math:`N` points per direction of an interval, square or cube of
side length :math:`L`. The grid is periodic (``bc='periodic'``, spacing :math:`h = L / N`) or has zero Dirichlet
values (``bc='dirichlet-zero'``: :math:`N` interior points of a vertex-centered grid, :math:`h = L / (N + 1)`). Both
are the grids, and :math:`A_h` the matrix, of pySDC's finite differences
(:func:`~pySDC.helpers.problem_helper.get_finite_difference_matrix` with ``order=2, stencil_type='center'`` for
:math:`\Delta_h` and ``order=1, stencil_type='upwind'`` for positive velocities). Without a velocity it is pure
diffusion, :math:`A_h = \nu \Delta_h`. :class:`GeometricMultigrid` takes and returns plain arrays and does not depend
on the rest of pySDC; :class:`~pySDC.implementations.problem_classes.generic_ND_FD.GenericNDimFinDiff` uses it for
``solver_type='MG'``.

**The method.** V(2,2)-cycles with weighted Jacobi smoothing (:math:`\omega = 0.8`, with the diagonal of the whole
operator), full-weighting restriction, linear interpolation, the same stencil on every grid, and an exact solve with a
dense inverse on the coarsest grid, which has at most ``coarsest`` points (default ``COARSEST``, 1000). A periodic
grid coarsens from :math:`N` to :math:`N / 2` points per direction, a Dirichlet grid from :math:`N` to
:math:`(N - 1) / 2` interior points, with zero boundary values in the transfers. The exact coarse solve is needed: the smooth modes of :math:`I - s \Delta_h` are
damped only by the identity part, which Jacobi does not do. A solve starts from zero and stops when the 2-norm of the
residual is below ``rtol`` times that of ``b``, or when a V-cycle no longer halves it (``stall``), which is where the
floating-point format of ``b`` ends (about :math:`\kappa u`, with :math:`\kappa` the condition number and :math:`u`
the unit roundoff), or after ``maxiter`` cycles. A cycle that does not reduce the residual at all is discarded, so
that a format too coarse for the system returns its best iterate, zero in the worst case, rather than one that is
worse.

**Advection.** Point Jacobi smooths reliably while diffusion dominates on every grid, and the grid Peclet number
:math:`|c_a| H / \nu` is largest on the coarsest grid actually used (spacing :math:`H`, set by ``coarsest``), which is
what bounds it; at ``coarsest=15``, a 1D grid has :math:`H = 1/16` whatever :math:`N`. Measured with random
right-hand sides in 1D to 3D, periodic and Dirichlet, with stiffness :math:`s \nu h^{-2}` up to :math:`10^6`, every
solve reaches ``rtol`` while that Peclet number is at most about 1 to 2. Beyond, a V-cycle reduces the residual by
only 0.5 to 0.85 in 2D and 3D (the coarse grids, upwinded on their own spacing, approximate the smooth components
along the flow badly), the stall rule takes that for the end of the format, and the solve stops after a few cycles
with a residual of :math:`10^{-6}` to :math:`10^{-2}`.

The first V-cycle is a second, separate trap, in 1D as in 2D: for a smooth right-hand side and a stiff system
(:math:`s \nu h^{-2} \gtrsim 10^4`) with advection, the first cycle from zero leaves its error in the outflow
boundary layer, where the residual is large, and reduces the 2-norm of the residual by only 0.5 to 0.9, or increases
it (up to four times measured), even at coarsest-grid Peclet numbers far below 1 (0.03 at :math:`N = 1023` and
:math:`H = 1/512`). The cycles after it contract by 0.15 to 0.7. The stall rule then ends the solve after the first
cycle, or, if the residual grew, the solve discards it and returns zero. Raising ``stall`` (to 0.8 or 0.9) lets
slowly contracting solves go on to ``rtol`` (in 35 V-cycles at a coarsest-grid Peclet number of 16 in 2D, at the
price of about one more cycle at the end of a low format), and is needed near the limit; it does not help where the
first cycle increases the residual. Every solve leaves the relative residual of what it returns in
``relative_residual``: check it with advection.

**Formats.** A solve computes in the floating-point type of its right-hand side: ``float64`` and ``float32`` arrays
are solved in their own arithmetic, and ``float16`` is a storage format, as on GPUs: every array is stored in half
precision, every operation computes in single precision. The result has the type of the right-hand side.

**Speed.** On a GPU, periodic 3D grids run on fused CUDA kernels: one pass over the arrays for the residual, the
Jacobi step, the restriction and the interpolation each, with the grid size compiled in. With ``fma=False`` (default), the
stencil kernels compute bit for bit what the array operations compute, which is what the tests check; ``fma=True``
lets the compiler contract multiplications and additions, which is faster where the arithmetic and not the memory
limits the kernels (fp64 on GPUs with few fp64 units). Measured at :math:`640^3` with V(2,2)-cycles, one Jacobi step
moves data at about 3 TB/s on an H100 in fp64. The fused kernels are for periodic pure diffusion; 1D and 2D grids,
Dirichlet grids, advection, and every grid on a CPU use array operations.
"""

import re

import numpy as np

OMEGA = 0.8  # Jacobi weight
COARSEST = 1000  # default: grids with at most this many points are solved exactly
STALL = 0.5  # default: a V-cycle that reduces the residual by less than this is at the accuracy of the format


def compute_type(dtype):
    """The type a format computes in: its own, but float32 for float16, which is a storage format here."""
    return np.promote_types(dtype, np.float32)


class _Ops:
    """
    The multigrid operations on one backend ``xp``: array operations everywhere, fused CuPy kernels for 3D grids on a
    GPU. Both compute in :func:`compute_type` and store in the type of the input.
    """

    # the CUDA types of the stored arrays and of their arithmetic
    _TYPES = {
        np.dtype(np.float64): ('double', 'double'),
        np.dtype(np.float32): ('float', 'float'),
        np.dtype(np.float16): ('__half', 'float'),
    }

    def __init__(self, xp, fma=False, periodic=True):
        self.xp = xp
        self.gpu = xp is not np
        self.periodic = periodic
        self.start = 0 if periodic else 1  # the fine index of the first coarse point
        # off, the kernels compute bit for bit what the array operations do; on, they are faster where arithmetic
        # limits them
        self.options = () if fma else ('--fmad=false',)
        self._compiled = {}
        # planes per thread in the streaming stencils: on an H100 at 640^3, 16 to 64 are equally fast, longer sections
        # lose up to 25 % (too few thread blocks for even waves), shorter ones re-read more planes
        self.section = 64

    def fused(self, x):
        """Whether ``x`` is handled by the fused kernels: a periodic 3D grid of a supported type on a GPU."""
        return self.gpu and self.periodic and x.ndim == 3 and x.dtype in self._TYPES

    @staticmethod
    def omega_over_d(x, sh, q=None):
        r"""
        :math:`\omega / (1 + 2 d\, s \nu h^{-2} + \sum_a |s c_a h^{-1}|)`, the Jacobi factor for ``d``-dimensional
        ``x``, with ``sh`` :math:`= s \nu h^{-2}` and ``q`` the :math:`s c_a h^{-1}`, computed once on the host in the
        compute type: no division per point, which costs real time where fp64 arithmetic is slow.
        """
        ct = compute_type(x.dtype).type
        diagonal = ct(1) + ct(2 * x.ndim) * ct(sh)
        if q:
            diagonal = diagonal + ct(sum(abs(v) for v in q))
        return ct(OMEGA) / diagonal

    def shift(self, y, k, axis):
        """``y`` moved by ``k`` = 1 or -1 along ``axis`` as by ``numpy.roll``: periodic, or with zero boundary values."""
        if self.periodic:
            return self.xp.roll(y, k, axis)
        out = self.xp.zeros_like(y)
        inner, before = (slice(1, None), slice(None, -1)), (slice(None),) * axis
        destination, source = inner if k == 1 else inner[::-1]
        out[before + (destination,)] = y[before + (source,)]
        return out

    # --- the operations ---------------------------------------------------------------------------------------------
    def stencil(self, kind, x, b, p1, p2=0.0, out=None, q=None):
        r"""
        A stencil operation with :math:`B x = p_1 \Delta x - \sum_a q_a D_a x`, ``p1`` the scale of the Laplacian
        (unscaled :math:`\Delta`) and ``q`` (or None) the signed scales of the first-order upwind differences
        :math:`D_a`: 0 gives :math:`B x`, 1 the residual :math:`b - (x - B x)`, 2 the Jacobi step
        :math:`x + p_2 (b - (x - B x))`, with ``p2`` from :meth:`omega_over_d`.
        """
        if q is None and self.fused(x):
            return self._stencil_fused(kind, x, b, p1, p2, out)
        ct = compute_type(x.dtype)
        y = x.astype(ct, copy=False)
        s = sum(self.shift(y, shift, axis) for axis in range(y.ndim) for shift in (1, -1))
        lap = s - (2 * y.ndim) * y
        result = ct.type(p1) * lap
        for axis, qa in enumerate(q or ()):
            if qa:  # q_a D_a x = |q_a| (x_i - x_upwind), the upwind neighbour on the side the velocity comes from
                result = result - ct.type(abs(qa)) * (y - self.shift(y, 1 if qa > 0 else -1, axis))
        if kind != 0:
            result = b.astype(ct, copy=False) - (y - result)
            if kind == 2:
                result = y + ct.type(p2) * result
        if out is None:
            return result.astype(x.dtype, copy=False)
        out[...] = result
        return out

    def residual(self, x, b, s, h2inv, q=None):
        return self.stencil(1, x, b, s * h2inv, q=q)

    def jacobi(self, x, b, s, h2inv, q=None):
        """One weighted Jacobi step, out of place."""
        return self.stencil(2, x, b, s * h2inv, self.omega_over_d(x, s * h2inv, q), q=q)

    def jacobi_from_zero(self, b, s, h2inv, q=None):
        """The Jacobi step from x = 0, which is (omega / d) b: one pass over b instead of a stencil over zeros."""
        wd = self.omega_over_d(b, s * h2inv, q)
        if self.fused(b):
            return self._kernel('jacobi_zero', b)(b, self._scalar(b, wd), self.xp.empty_like(b))
        return (wd * b.astype(compute_type(b.dtype), copy=False)).astype(b.dtype, copy=False)

    def restrict(self, r):
        """Full weighting, [1/4, 1/2, 1/4] in every direction, onto every second point."""
        if self.fused(r):
            n = r.shape[0] // 2
            return self._kernel('restrict', r, n)(r, self.xp.empty((n, n, n), dtype=r.dtype), size=n**3)
        y = r.astype(compute_type(r.dtype), copy=False)
        for axis in range(r.ndim):
            y = 0.25 * self.shift(y, 1, axis) + 0.5 * y + 0.25 * self.shift(y, -1, axis)
        return y[(slice(self.start, None, 2),) * r.ndim].astype(r.dtype)

    def prolong_add(self, x, e):
        """x += linear interpolation of the coarse correction e, in place."""
        if self.fused(x):
            self._kernel('prolong', x)(e, x, size=e.size)
            return x
        fine = self.xp.zeros(x.shape, dtype=compute_type(x.dtype))
        fine[(slice(self.start, None, 2),) * x.ndim] = e
        for axis in range(x.ndim):
            fine = 0.5 * self.shift(fine, 1, axis) + fine + 0.5 * self.shift(fine, -1, axis)
        x[...] = x.astype(fine.dtype, copy=False) + fine
        return x

    # --- fused CuPy kernels for 3D grids ----------------------------------------------------------------------------
    @staticmethod
    def _scalar(x, value):
        """A kernel scalar for arrays like x: of their compute type."""
        return compute_type(x.dtype).type(value)

    _STENCIL = r"""
    #include <cuda_fp16.h>
    template <typename CT, typename T> __device__ __forceinline__ CT ld(const T* p, int i) { return CT(p[i]); }
    template <> __device__ __forceinline__ float ld<float, __half>(const __half* p, int i) {
      return __half2float(p[i]);
    }
    template <typename T, typename CT> __device__ __forceinline__ T st(CT v) { return T(v); }
    template <> __device__ __forceinline__ __half st<__half, float>(float v) { return __float2half_rn(v); }

    // KIND 0: y = p1 lap x; 1: residual y = bb - (x - p1 lap x); 2: Jacobi step with p2 = omega / d.
    // Every thread owns one column (b, c) of a section of L_ planes and walks along a, keeping the planes a - 1, a,
    // a + 1 in registers, so that x is read about once from memory (twice at the ends of a section); the sections
    // give the GPU enough threads to keep enough loads in flight. The operations are those of the array version, in
    // the same order.
    template <typename T, typename CT, int KIND>
    __global__ void stencil(const T* __restrict__ x, const T* __restrict__ bb, T* __restrict__ y, CT p1, CT p2) {
      const int n = N_, nn = n * n;
      const int c = blockIdx.x * blockDim.x + threadIdx.x, b = blockIdx.y * blockDim.y + threadIdx.y;
      if (b >= n || c >= n) return;
      const int rest = b * n + c;
      const int bm = (b == 0 ? n - 1 : b - 1) * n + c, bp = (b == n - 1 ? 0 : b + 1) * n + c;
      const int cm = b * n + (c == 0 ? n - 1 : c - 1), cp = b * n + (c == n - 1 ? 0 : c + 1);
      const int first = blockIdx.z * L_, last = min(first + L_, n);
      CT prev = ld<CT>(x, (first == 0 ? n - 1 : first - 1) * nn + rest), cur = ld<CT>(x, first * nn + rest);
      for (int a = first; a < last; a++) {
        const int plane = a * nn;
        const CT next = ld<CT>(x, (a == n - 1 ? 0 : plane + nn) + rest);
        CT s = prev + next;
        s = s + ld<CT>(x, plane + bm);
        s = s + ld<CT>(x, plane + bp);
        s = s + ld<CT>(x, plane + cm);
        s = s + ld<CT>(x, plane + cp);
        const CT xk = cur;
        const CT lap = s - CT(6) * xk;
        const int k = plane + rest;
        if (KIND == 0) {
          y[k] = st<T>(lap * p1);
        } else if (KIND == 1) {
          y[k] = st<T>(ld<CT>(bb, k) - (xk - p1 * lap));
        } else {
          y[k] = st<T>(xk + p2 * (ld<CT>(bb, k) - (xk - p1 * lap)));
        }
        prev = cur;
        cur = next;
      }
    }
    """

    def _stencil_fused(self, kind, x, b, p1, p2, out):
        """
        The stencil on a GPU, in its fastest form for the type, measured on an H100 at 640^3: one thread per point
        for fp64 (2.1 ms for a Jacobi step, against 2.8 ms streaming), streaming columns for fp32 (1.24 against
        1.38 ms) and fp16 (0.86 against 1.28 ms). Both forms compute the same, bit for bit.
        """
        n = x.shape[0]
        out = self.xp.empty_like(x) if out is None else out
        if x.dtype == np.float64:
            name = ('lap', 'residual', 'jacobi')[kind]
            scalars = (self._scalar(x, p1),) if kind != 2 else (self._scalar(x, p1), self._scalar(x, p2))
            return self._kernel(name, x)(x, *(() if kind == 0 else (b,)), *scalars, out)
        section = min(n, self.section)
        if ('stencil', n, section) not in self._compiled:
            expressions = [f'stencil<{t},{ct},{k}>' for t, ct in self._TYPES.values() for k in range(3)]
            self._compiled['stencil', n, section] = self.xp.RawModule(
                code=self._STENCIL,
                options=(*self.options, f'-DN_={n}', f'-DL_={section}'),
                name_expressions=expressions,
            )
        t, ct = self._TYPES[x.dtype]
        kernel = self._compiled['stencil', n, section].get_function(f'stencil<{t},{ct},{kind}>')
        threads = (32, 8)
        blocks = ((n + threads[0] - 1) // threads[0], (n + threads[1] - 1) // threads[1], (n + section - 1) // section)
        kernel(blocks, threads, (x, x if b is None else b, out, self._scalar(x, p1), self._scalar(x, p2)))
        return out

    def _kernel(self, name, x, n=None):
        r"""
        The fused kernel ``name`` for arrays like ``x`` on grids of :math:`n^3` points (default: the size of ``x``),
        compiled once per grid size and type.

        The grid size is a compile-time constant: the index arithmetic then divides by constants, which the compiler
        turns into multiplications, instead of by a size read at run time, which is slow on a GPU and does not shrink
        with the format, so that it would bound a kernel in fp32 rather than its memory traffic. Arithmetic is in the
        type ``CT``: the array's own type, but float32 for float16.
        """
        n = x.shape[0] if n is None else n
        half = x.dtype == np.float16
        if (name, n, half) not in self._compiled:
            neighbours = """
            const int n = N_, nn = n * n;
            int k = i, a = k / nn, b = (k / n) % n, c = k % n;
            int am = (a == 0 ? n - 1 : a - 1) * nn, ap = (a == n - 1 ? 0 : a + 1) * nn;
            int bm = (b == 0 ? n - 1 : b - 1) * n, bp = (b == n - 1 ? 0 : b + 1) * n;
            int cm = c == 0 ? n - 1 : c - 1, cp = c == n - 1 ? 0 : c + 1;
            int rest = b * n + c, plane = a * nn;
            CT s = CT(x[am + rest]) + CT(x[ap + rest]);
            s = s + CT(x[plane + bm + c]);
            s = s + CT(x[plane + bp + c]);
            s = s + CT(x[plane + b * n + cm]);
            s = s + CT(x[plane + b * n + cp]);
            CT xk = CT(x[k]);
            CT lap = s - CT(6) * xk;
            """
            sources = {
                # one thread per point: the fastest in fp64, which reaches the memory bandwidth this way
                'lap': ('raw T x, S h2inv', 'T y', neighbours + 'y = T(lap * h2inv)'),
                'residual': ('raw T x, T bb, S sh', 'T r', neighbours + 'r = T(CT(bb) - (xk - sh * lap))'),
                'jacobi': (
                    'raw T x, T bb, S sh, S wd',
                    'T y',
                    neighbours + 'y = T(xk + wd * (CT(bb) - (xk - sh * lap)))',
                ),
                'jacobi_zero': ('T bb, S wd', 'T y', 'y = T(wd * CT(bb));'),
                'restrict': (
                    'raw T r',
                    'raw T out',
                    """
                    const int nc = N_, n = 2 * nc;
                    int k = i, a = k / (nc * nc), b = (k / nc) % nc, c = k % nc;
                    CT acc = CT(0);
                    for (int da = -1; da <= 1; da++) {
                      int fa = (2 * a + da + n) % n; CT wa = da == 0 ? CT(0.5) : CT(0.25);
                      for (int db = -1; db <= 1; db++) {
                        int fb = (2 * b + db + n) % n; CT wb = db == 0 ? CT(0.5) : CT(0.25);
                        for (int dc = -1; dc <= 1; dc++) {
                          int fc = (2 * c + dc + n) % n; CT wc = dc == 0 ? CT(0.5) : CT(0.25);
                          acc = acc + (wa * wb * wc) * CT(r[(fa * n + fb) * n + fc]);
                        }
                      }
                    }
                    out[k] = T(acc);
                    """,
                ),
                'prolong': (
                    'raw T e',
                    'raw T x',
                    """
                    // one thread per coarse point (A, B, C): it reads the eight coarse values of its cell once and
                    // adds the interpolation to the eight fine points (2A + da, 2B + db, 2C + dc), with all eight
                    // terms in a fixed order, those absent at its parity with weight zero
                    const int n = N_, nc = n / 2;
                    int k = i, A = k / (nc * nc), B = (k / nc) % nc, C = k % nc;
                    int A1 = A == nc - 1 ? 0 : A + 1, B1 = B == nc - 1 ? 0 : B + 1, C1 = C == nc - 1 ? 0 : C + 1;
                    CT e000 = CT(e[(A * nc + B) * nc + C]), e001 = CT(e[(A * nc + B) * nc + C1]);
                    CT e010 = CT(e[(A * nc + B1) * nc + C]), e011 = CT(e[(A * nc + B1) * nc + C1]);
                    CT e100 = CT(e[(A1 * nc + B) * nc + C]), e101 = CT(e[(A1 * nc + B) * nc + C1]);
                    CT e110 = CT(e[(A1 * nc + B1) * nc + C]), e111 = CT(e[(A1 * nc + B1) * nc + C1]);
                    #pragma unroll
                    for (int da = 0; da < 2; da++) {
                      #pragma unroll
                      for (int db = 0; db < 2; db++) {
                        #pragma unroll
                        for (int dc = 0; dc < 2; dc++) {
                          CT wa = da ? CT(0.5) : CT(1), wb = db ? CT(0.5) : CT(1), wc = dc ? CT(0.5) : CT(1);
                          CT ma = da ? CT(1) : CT(0), mb = db ? CT(1) : CT(0), mc = dc ? CT(1) : CT(0);
                          CT w = wa * wb * wc, v = CT(0);
                          v = v + (w * CT(1)) * e000;
                          v = v + (w * mc) * e001;
                          v = v + (w * mb) * e010;
                          v = v + (w * (mb * mc)) * e011;
                          v = v + (w * ma) * e100;
                          v = v + (w * (ma * mc)) * e101;
                          v = v + (w * (ma * mb)) * e110;
                          v = v + (w * (ma * mb * mc)) * e111;
                          int f = ((2 * A + da) * n + 2 * B + db) * n + 2 * C + dc;
                          x[f] = T(CT(x[f]) + v);
                        }
                      }
                    }
                    """,
                ),
            }
            params, out, body = sources[name]
            # the scalars in the compute type: CuPy infers type placeholders from arrays only
            params = re.sub(r'\bS\b', 'float32' if half else 'T', params)
            self._compiled[name, n, half] = self.xp.ElementwiseKernel(
                params,
                out,
                body.replace('CT', 'float' if half else 'T'),
                f'mg_{name}_{n}_{"h" if half else "f"}',
                options=self.options,
                preamble=f'#define N_ {n}',
            )
        return self._compiled[name, n, half]


class GeometricMultigrid:
    r"""
    Geometric multigrid for :math:`(I - s A_h)\, x = b` with :math:`A_h = \nu \Delta_h - c \cdot \nabla_h` on a grid
    of :math:`N^d` points with side length :math:`L`, periodic or with zero Dirichlet values: :math:`\Delta_h` is the
    second-order Laplacian and :math:`\nabla_h` the first-order upwind gradient. See :mod:`pySDC.helpers.multigrid`
    for the method, the formats and the speed.

    Parameters
    ----------
    N : int
        Points per direction. Periodic: :math:`N = m 2^k` with :math:`m^d` at most ``coarsest``; zero Dirichlet: the
        interior points, :math:`N = 2^k - 1` (or :math:`m 2^k - 1` with :math:`(m - 1)^d` at most ``coarsest``).
    ndim : int
        Dimension: 1, 2 or 3.
    L : float
        Side length of the domain.
    xp : module
        ``numpy`` or ``cupy``: where the arrays live.
    fma : bool
        On a GPU, let the compiler contract multiplications and additions into fused multiply-adds: faster where
        arithmetic limits the kernels, but no longer bit for bit what the array operations compute.
    bc : str
        ``'periodic'`` (grid spacing :math:`h = L / N`) or ``'dirichlet-zero'``: :math:`N` interior points of a
        vertex-centered grid with :math:`h = L / (N + 1)` and zero values on the boundary.
    nu : float
        Diffusion coefficient :math:`\nu`, positive.
    velocity : sequence of float or None
        Constant velocity :math:`c`, one entry per dimension; None or zeros give pure diffusion. Each component is
        upwinded on the side it comes from.
    coarsest : int
        The largest number of points the coarsest grid may have, which is solved exactly with a dense inverse: the grid
        is halved until it has at most this many. Default ``COARSEST`` (1000); smaller values give more levels, and
        a coarsest grid with a larger spacing, which is what bounds the Peclet number (see
        :mod:`pySDC.helpers.multigrid`).
    stall : float
        The stall rule: a V-cycle that reduces the residual by a factor larger than this ends the solve, taken for
        the accuracy of the format. Raise it (to 0.9, say) where V-cycles contract slowly, as on advection-dominated
        coarse grids, so that they go on to ``rtol``; the price is more cycles spent at the accuracy of a low format
        (fp16, fp32) before the solve notices it.

    Attributes
    ----------
    relative_residual : float
        The 2-norm of the residual of the iterate the last :meth:`solve` returned, relative to that of ``b``.

    Examples
    --------
    >>> import numpy as np
    >>> mg = GeometricMultigrid(64, ndim=2)
    >>> b = np.random.default_rng(0).standard_normal((64, 64))
    >>> x, cycles = mg.solve(b, s=1e-2, rtol=1e-10, maxiter=50)
    >>> bool(np.linalg.norm(b - (x - mg.apply(x, 1e-2))) < 1e-10 * np.linalg.norm(b))
    True
    >>> mg = GeometricMultigrid(63, ndim=2, bc='dirichlet-zero', nu=0.1, velocity=(1.0, -0.5))
    >>> b = np.random.default_rng(0).standard_normal((63, 63))
    >>> x, cycles = mg.solve(b, s=1e-1, rtol=1e-10, maxiter=50)
    >>> bool(np.linalg.norm(b - (x - mg.apply(x, 1e-1))) < 1e-10 * np.linalg.norm(b))
    True
    """

    def __init__(
        self, N, ndim=3, L=1.0, xp=np, fma=False, bc='periodic', nu=1.0, velocity=None, stall=STALL, coarsest=COARSEST
    ):
        if ndim not in (1, 2, 3):
            raise ValueError(f'ndim has to be 1, 2 or 3, got {ndim}')
        if bc not in ('periodic', 'dirichlet-zero'):
            raise ValueError(f'bc has to be "periodic" or "dirichlet-zero", got {bc}')
        if velocity is not None:
            velocity = tuple(float(c) for c in velocity)
            if len(velocity) != ndim:
                raise ValueError(f'velocity needs one entry per dimension, {ndim}, got {len(velocity)}')
            velocity = velocity if any(velocity) else None  # pure diffusion keeps the fused kernels
        periodic = bc == 'periodic'
        # halve the grid: n -> n / 2 on periodic grids, n -> (n - 1) / 2 interior points with Dirichlet values
        self.sizes = [N]
        while self.sizes[-1] ** ndim > coarsest and self.sizes[-1] % 2 == (0 if periodic else 1):
            self.sizes.append(self.sizes[-1] // 2)
        if self.sizes[-1] ** ndim > coarsest:
            form = f'm 2^k with m^{ndim}' if periodic else f'2^k - 1, or m 2^k - 1 with (m - 1)^{ndim}'
            raise ValueError(
                f'N = {N} does not coarsen to at most {coarsest} points in {ndim}D: use N = {form} <= {coarsest}'
            )
        self.N, self.ndim, self.L, self.xp, self.ops = N, ndim, L, xp, _Ops(xp, fma, periodic)
        self.bc, self.nu, self.velocity, self.stall = bc, nu, velocity, stall
        self.relative_residual = None
        self._coarse = {}

    def _scales(self, n, s):
        r"""
        For the grid of ``n`` points per direction: :math:`s \nu`, :math:`h^{-2}` and the signed
        :math:`s c_a h^{-1}` (None without advection), which is how :class:`_Ops` takes the operator.
        """
        hinv = (n if self.ops.periodic else n + 1) / self.L
        q = None if self.velocity is None else tuple(float(s * c * hinv) for c in self.velocity)
        return s * self.nu, float(hinv**2), q

    def solve(self, b, s, rtol, maxiter):
        r"""
        Solve :math:`(I - s A_h)\, x = b` from a zero guess, in the format of ``b``.

        Stops when the 2-norm of the residual is below ``rtol`` times that of ``b``, when a V-cycle reduces it by
        less than ``stall`` (the accuracy of the format), or after ``maxiter`` V-cycles. A solve that stops on the
        stall rule may not have reached ``rtol``: where V-cycles contract slowly, as beyond the Peclet limit (see
        :mod:`pySDC.helpers.multigrid`), the stall rule mistakes that for the end of the format. The relative
        residual of the returned iterate is in :attr:`relative_residual` after every solve.

        Parameters
        ----------
        b : array
            Right-hand side, of shape ``(N,) * ndim`` and type float64, float32 or float16.
        s : float
            Factor of the operator, positive.
        rtol : float
            Relative tolerance for the residual.
        maxiter : int
            Largest number of V-cycles.

        Returns
        -------
        x : array
            The solution, of the shape and type of ``b``.
        cycles : int
            The V-cycles it took.
        """
        if b.shape != (self.N,) * self.ndim:
            raise ValueError(f'expected a right-hand side of shape {(self.N,) * self.ndim}, got {b.shape}')
        norm_b = self._norm(b)
        if norm_b == 0:
            self.relative_residual = 0.0
            return self.xp.zeros_like(b), 0
        x, cycles, previous = None, 0, norm_b  # x = None is zero, which the first smoothing step knows without reading
        while cycles < maxiter:
            candidate = self._vcycle(0, x, b, s)
            cycles += 1
            norm_r = self._norm(self.ops.residual(candidate, b, *self._scales(self.N, s)))
            if norm_r >= previous:  # past the accuracy of the format: the cycle made it worse, keep what was there
                break
            x, stalled, previous = candidate, norm_r > self.stall * previous, norm_r
            # converged, or at the accuracy the format allows: a V-cycle otherwise reduces the residual by 0.1 to 0.3
            if norm_r <= rtol * norm_b or stalled:
                break
        self.relative_residual = previous / norm_b
        return (self.xp.zeros_like(b) if x is None else x), cycles

    def laplacian(self, x, scale=1.0, out=None):
        r"""
        :math:`\text{scale}\cdot\Delta_h x` on the finest grid, without :math:`\nu` and the advection, in the type
        of ``x``, into ``out`` if given; on a GPU on periodic 3D grids by the fused kernel.
        """
        return self.ops.stencil(0, x, None, scale * self._scales(self.N, 1.0)[1], out=out)

    def apply(self, x, scale=1.0, out=None):
        r""":math:`\text{scale}\cdot A_h x` on the finest grid, the operator the solve inverts, in the type of ``x``."""
        sn, h2inv, q = self._scales(self.N, scale)
        return self.ops.stencil(0, x, None, sn * h2inv, out=out, q=q)

    def _norm(self, v):
        """The 2-norm; on a GPU by cuBLAS, in one pass, where cupy.linalg.norm reaches a third of the bandwidth."""
        flat = v.reshape(-1).astype(compute_type(v.dtype), copy=False)
        if self.ops.gpu:
            from cupy import cublas

            return float(cublas.nrm2(flat))
        return float(np.linalg.norm(flat))

    def _vcycle(self, level, x, b, s):
        if level == len(self.sizes) - 1:
            return self._coarse_solve(b, s)
        scales = self._scales(self.sizes[level], s)
        x = self.ops.jacobi_from_zero(b, *scales) if x is None else self.ops.jacobi(x, b, *scales)
        x = self.ops.jacobi(x, b, *scales)
        coarse = self.ops.restrict(self.ops.residual(x, b, *scales))
        x = self.ops.prolong_add(x, self._vcycle(level + 1, None, coarse, s))
        for _ in range(2):
            x = self.ops.jacobi(x, b, *scales)
        return x

    def _coarse_solve(self, b, s):
        """Exact solve on the coarsest grid with the dense inverse of its operator, kept per s and type."""
        n, ct = self.sizes[-1], compute_type(b.dtype)
        key = (s, ct.str)
        if key not in self._coarse:
            if len(self._coarse) >= 12:  # an adaptive step size keeps changing s
                self._coarse.pop(next(iter(self._coarse)))
            shape, (sn, h2inv, q) = (n,) * self.ndim, self._scales(n, s)
            ops = _Ops(np, periodic=self.ops.periodic)
            columns = [
                e - ops.stencil(0, e.reshape(shape), None, sn * h2inv, q=q).reshape(-1) for e in np.eye(n**self.ndim)
            ]
            self._coarse[key] = self.xp.asarray(np.linalg.inv(np.array(columns).T)).astype(ct)
        inverse = self._coarse[key]
        return (inverse @ b.reshape(-1).astype(ct, copy=False)).astype(b.dtype).reshape(b.shape)
