#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 11 22:39:30 2023
"""

import numpy as np
import scipy.fft
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from pySDC.core.errors import ProblemError
from pySDC.core.problem import Problem, WorkCounter
from pySDC.helpers import problem_helper
from pySDC.implementations.datatype_classes.mesh import mesh


class GenericNDimFinDiff(Problem):
    r"""
    Base class for finite difference spatial discretisation in :math:`N` dimensions

    .. math::
        \frac{d u}{dt} = A u,

    where :math:`A \in \mathbb{R}^{nN \times nN}` is a matrix arising from finite difference discretisation of spatial
    derivatives with :math:`n` degrees of freedom per dimension and :math:`N` dimensions. This generic class follows the MOL
    (method-of-lines) approach and can be used to discretize partial differential equations such as the advection
    equation and the heat equation.

    Parameters
    ----------
    nvars : int, optional
        Spatial resolution for the ND problem. For :math:`N = 2`,
        set ``nvars=(16, 16)``.
    coeff : float, optional
        Factor for finite difference matrix :math:`A`.
    derivative : int, optional
        Order of the spatial derivative.
    freq : tuple of int, optional
        Spatial frequency, can be a tuple.
    stencil_type : str, optional
        Stencil type for finite differences.
    order : int, optional
        Order of accuracy of the finite difference discretization.
    lintol : float, optional
        Tolerance for spatial solver.
    liniter : int, optional
        Maximum number of iterations for linear solver.
    dtype : dtype-like, optional
        Precision the state is stored at. ``float64`` by default, which is what every caller got
        before this existed. The operators follow at ``promote_types(dtype, float32)``, since SciPy
        has no half-precision sparse matrix and hardware that stores half precision computes in
        single anyway -- so ``float16`` here means genuinely half-precision *storage* with
        single-precision arithmetic, which is the arrangement it has on real hardware too.
    solver_type : str, optional
        Type of solver. Can be ``'direct'``, ``'GMRES'``, ``'CG'`` or, on periodic grids, ``'FFT'``:
        the operator is then circulant, so the FFT diagonalises it and a solve is two transforms
        and a division -- exact, and the cheapest solve there is on a GPU.
    solve_precision : dtype-like or None, optional
        Precision the implicit solve runs at, independently of the state's ``dtype``. ``None``
        (default) solves at the operator's precision, exactly as without it. Otherwise the solve
        genuinely works in that precision: the sparse solvers on a copy of the operator held at it,
        ``'FFT'`` with transforms at it, and the result is widened back to ``dtype``. Sparse
        matrices stop at single precision in SciPy and CuPy alike, so ``float16`` needs ``'FFT'``.
        On a GPU those are cuFFT's half-precision transforms (power-of-two grids), with every value
        stored and every product rounded in half precision. A CPU has no half-precision FFT, so
        there ``float16`` is *emulated*, by rounding through it around a single-precision
        transform -- which is optimistic, since the transform's own arithmetic is then single.
        For a complex state the solve is complex at the same precision.
    bc : str or tuple of 2 string, optional
        Type of boundary conditions. Default is ``'periodic'``.
        To define two different types of boundary condition for each side,
        you can use a tuple, for instance ``bc=("dirichlet", "neumann")``
        uses Dirichlet BC on the left side, and Neumann BC on the right side.
    bcParams : dict, optional
        Parameters for boundary conditions, that can contains those keys :

        - **val** : value for the boundary value (Dirichlet) or derivative
          (Neumann), default to 0
        - **reduce** : if true, reduce the order of the A matrix close to the
          boundary. If false (default), use shifted stencils close to the
          boundary.
        - **neumann_bc_order** : finite difference order that should be used
          for the neumann BC derivative. If None (default), uses the same
          order as the discretization for A.

        Default is None, which takes the default values for each parameters.
        You can also define a tuple to set different parameters for each
        side.
    useGPU : bool, optional
        Run on the GPU with CuPy instead of on the CPU with NumPy.

    Attributes
    ----------
    A : sparse matrix (CSC)
        FD discretization matrix of the ND operator.
    Id : sparse matrix (CSC)
        Identity matrix of the same dimension as A.
    xvalues : np.1darray
        Values of spatial grid.
    """

    dtype_u = mesh
    dtype_f = mesh
    xp = np
    xsp = sp
    linalg = spla

    def setup_GPU(self):
        """
        Switch to GPU modules
        """
        import cupy as cp
        import cupyx.scipy.sparse as csp
        import cupyx.scipy.sparse.linalg as cspla

        from pySDC.implementations.datatype_classes.cupy_mesh import cupy_mesh

        self.xp = cp
        self.xsp = csp
        self.linalg = cspla
        self.dtype_u = cupy_mesh
        self.dtype_f = cupy_mesh

    def __init__(
        self,
        nvars=512,
        coeff=1.0,
        derivative=1,
        freq=2,
        stencil_type='center',
        order=2,
        lintol=1e-12,
        liniter=10000,
        solver_type='direct',
        bc='periodic',
        bcParams=None,
        dtype='float64',
        useGPU=False,
        solve_precision=None,
    ):
        """Initialization routine"""
        if useGPU:
            self.setup_GPU()

        # make sure parameters have the correct types
        if type(nvars) not in [int, tuple]:
            raise ProblemError('nvars should be either tuple or int')
        if type(freq) not in [int, tuple]:
            raise ProblemError('freq should be either tuple or int')

        # transforms nvars into a tuple
        if type(nvars) is int:
            nvars = (nvars,)

        # automatically determine ndim from nvars
        ndim = len(nvars)
        if ndim > 3:
            raise ProblemError(f'can work with up to three dimensions, got {ndim}')

        # eventually extend freq to other dimension
        if type(freq) is int:
            freq = (freq,) * ndim
        if len(freq) != ndim:
            raise ProblemError(f'len(freq)={len(freq)}, different to ndim={ndim}')

        # check values for freq and nvars
        for f in freq:
            if ndim == 1 and f == -1:
                # use Gaussian initial solution in 1D
                bc = 'periodic'
                break
            if f % 2 != 0 and bc == 'periodic':
                raise ProblemError('need even number of frequencies due to periodic BCs')
        for nvar in nvars:
            if nvar % 2 != 0 and bc == 'periodic':
                raise ProblemError('the setup requires nvars = 2^p per dimension')
            if (nvar + 1) % 2 != 0 and bc == 'dirichlet-zero':
                raise ProblemError('setup requires nvars = 2^p - 1')
        if ndim > 1 and nvars[1:] != nvars[:-1]:
            raise ProblemError('need a square domain, got %s' % nvars)

        # invoke super init, passing number of dofs and the precision to store them at
        dtype = np.dtype(dtype)

        # SciPy holds no float16 sparse matrix, and hardware that stores half precision computes in
        # single anyway, so the operators sit at the smallest single-or-wider type that holds `dtype`
        operator_dtype = np.promote_types(dtype, np.float32)

        super().__init__(init=(nvars[0] if ndim == 1 else nvars, None, dtype))

        dx, xvalues = problem_helper.get_1d_grid(size=nvars[0], bc=bc, left_boundary=0.0, right_boundary=1.0)

        self.A, _ = problem_helper.get_finite_difference_matrix(
            derivative=derivative,
            order=order,
            stencil_type=stencil_type,
            dx=dx,
            size=nvars[0],
            dim=ndim,
            bc=bc,
            cupy=useGPU,
        )
        self.A *= coeff

        self.A = self.A.astype(operator_dtype)

        # SciPy's sparse direct solver wants CSC and CuPy's wants CSR. Whichever one is handed the
        # wrong layout converts the whole matrix on every call -- that is what cupyx's
        # `SparseEfficiencyWarning: CSR format is required` is reporting, once per solve. CSR is
        # also the better layout for the matrix-vector product in `eval_f`.
        self.A = self.A.tocsr() if useGPU else self.A.tocsc()

        # the grid feeds every `u_exact`, so it has to live where the solution does
        self.xvalues = self.xp.asarray(xvalues)
        self.Id = self.xsp.eye(np.prod(nvars), format='csr' if useGPU else 'csc', dtype=operator_dtype)

        # store attribute and register them as parameters
        self._makeAttributeAndRegister('nvars', 'stencil_type', 'order', 'bc', localVars=locals(), readOnly=True)
        self._makeAttributeAndRegister('freq', 'lintol', 'liniter', 'solver_type', localVars=locals())
        self.dtype = dtype
        self.operator_dtype = operator_dtype

        if self.solver_type != 'direct':
            self.work_counters[self.solver_type] = WorkCounter()

        solve_precision = None if solve_precision is None else np.dtype(solve_precision)
        self._makeAttributeAndRegister('solve_precision', localVars=locals(), readOnly=True)
        self._setup_solve(solver_type, solve_precision, bc)

    def _setup_solve(self, solver_type, solve_precision, bc):
        """Prepare the reduced-precision operator, or the eigenvalues the FFT solve divides by."""
        # half precision is the one format that is not a complex dtype for a complex state
        self._half = solve_precision == np.float16
        precision = self.operator_dtype if solve_precision is None or self._half else solve_precision
        if self.dtype.kind == 'c':
            precision = np.promote_types(precision, np.complex64)
        self._solve_dtype = precision

        if solver_type == 'FFT':
            if bc != 'periodic':
                raise ProblemError(f'the FFT solve diagonalises a circulant operator and needs periodic BCs, got {bc}')
            if self._half and self.xp is not np and any(n & (n - 1) for n in self.nvars):
                raise ProblemError(f'cuFFT computes in half precision on power-of-two grids only, got {self.nvars}')
            self._fft = scipy.fft if self.xp is np else self.xp.fft
            # the first column of a circulant transforms to its eigenvalues
            first_column = self.xp.zeros(self.A.shape[0], dtype=self.operator_dtype)
            first_column[0] = 1.0
            self._eigenvalues = self._fft.fftn((self.A @ first_column).reshape(self.nvars))
            self._half_plan = None
        elif solve_precision is not None:
            if self._half:
                raise ProblemError('sparse matrices stop at single precision; float16 needs solver_type="FFT"')
            self._A_solve = self.A.astype(precision)
            self._Id_solve = self.Id.astype(precision)

    @property
    def ndim(self):
        """Number of dimensions of the spatial problem"""
        return len(self.nvars)

    @property
    def dx(self):
        """Size of the mesh (in all dimensions)"""
        return self.xvalues[1] - self.xvalues[0]

    @property
    def grids(self):
        """ND grids associated to the problem"""
        x = self.xvalues
        if self.ndim == 1:
            return x
        if self.ndim == 2:
            return x[None, :], x[:, None]
        if self.ndim == 3:
            return x[None, :, None], x[:, None, None], x[None, None, :]

    @classmethod
    def get_default_sweeper_class(cls):
        """
        Default sweeper for these problems, which are treated fully implicitly.

        Returns
        -------
        type
            The sweeper class ``generic_implicit``.
        """
        from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit

        return generic_implicit

    def eval_f(self, u, t):
        """
        Routine to evaluate the right-hand side of the problem.

        Parameters
        ----------
        u : dtype_u
            Current values.
        t : float
            Current time.

        Returns
        -------
        f : dtype_f
            Values of the right-hand side of the problem.
        """
        f = self.f_init
        f[:] = self.A.dot(u.flatten()).reshape(self.nvars)
        return f

    def eval_f_increment(self, base, delta, t):
        r"""
        Evaluate :math:`f(w + \delta) - f(w) = A\delta`, which carries an explicit factor
        :math:`\delta`.

        The operator is linear, so the increment is the operator applied to the correction and the
        base state does not enter. Supplying it means a sweeper never has to form the increment by
        subtracting two stored right-hand sides, whose cancellation error carries
        :math:`\varepsilon\|A\|` -- see :class:`pySDC.core.problem.Problem`.

        Parameters
        ----------
        base : dtype_u
            The base state, unused for a linear operator.
        delta : dtype_u
            The correction.
        t : float
            Current time, accepted for interface compatibility.

        Returns
        -------
        f : dtype_f
            The increment.
        """
        f = self.dtype_f(self.init)
        f[:] = self.A.dot(delta.flatten()).reshape(self.nvars)
        return f

    def solve_system(self, rhs, factor, u0, t):
        r"""
        Simple linear solver for :math:`(I-factor\cdot A)\vec{u}=\vec{rhs}`.

        Parameters
        ----------
        rhs : dtype_f
            Right-hand side for the linear system.
        factor : float
            Abbrev. for the local stepsize (or any other factor required).
        u0 : dtype_u
            Initial guess for the iterative solver.
        t : float
            Current time (e.g. for time-dependent BCs).

        Returns
        -------
        sol : dtype_u
            The solution of the linear solver.
        """
        if self.solver_type == 'FFT' or self.solve_precision is not None:
            return self._solve_at_precision(rhs, factor, u0)

        solver_type, Id, A, nvars, lintol, liniter, sol = (
            self.solver_type,
            self.Id,
            self.A,
            self.nvars,
            self.lintol,
            self.liniter,
            self.u_init,
        )

        if solver_type == 'direct':
            sol[:] = self.linalg.spsolve(Id - factor * A, rhs.flatten()).reshape(nvars)
        elif solver_type == 'GMRES':
            sol[:] = self.linalg.gmres(
                Id - factor * A,
                rhs.flatten(),
                x0=u0.flatten(),
                rtol=lintol,
                maxiter=liniter,
                atol=0,
                callback=self.work_counters[solver_type],
                callback_type='legacy',
            )[0].reshape(nvars)
        elif solver_type == 'CG':
            sol[:] = self.linalg.cg(
                Id - factor * A,
                rhs.flatten(),
                x0=u0.flatten(),
                rtol=lintol,
                maxiter=liniter,
                atol=0,
                callback=self.work_counters[solver_type],
            )[0].reshape(nvars)
        else:
            raise ValueError(f'solver type "{solver_type}" not known in generic advection-diffusion implementation!')

        return sol

    def _solve_at_precision(self, rhs, factor, u0):
        r"""
        Solve :math:`(I-factor\cdot A)\vec{u}=\vec{rhs}` at ``solve_precision``, or by FFT.

        Returns
        -------
        sol : dtype_u
            The solution, widened to the state's precision.
        """
        sol = self.dtype_u(self.init)
        if self.solver_type == 'FFT':
            sol[:] = self._solve_fft(rhs.view(self.xp.ndarray).reshape(self.nvars), factor)
            self.work_counters['FFT']()
            return sol

        precision = self._solve_dtype
        # the factor at the solve's own precision: a float64 scalar would widen the operator back
        matrix = self._Id_solve - precision.type(factor) * self._A_solve
        b = rhs.view(self.xp.ndarray).flatten().astype(precision)
        if self.solver_type == 'direct':
            x = self.linalg.spsolve(matrix, b)
        elif self.solver_type in ['GMRES', 'CG']:
            x0 = u0.view(self.xp.ndarray).flatten().astype(precision)
            options = {'callback_type': 'legacy'} if self.solver_type == 'GMRES' else {}
            # A Krylov solve cannot meet a tolerance below its own precision. Asked to, it iterates to
            # `liniter` and, in CG, divides by a dot product that has rounded to zero: NaN. A hundred
            # times the precision's epsilon is what single precision still reaches.
            rtol = max(self.lintol, 100 * float(np.finfo(precision).eps))
            x = getattr(self.linalg, self.solver_type.lower())(
                matrix,
                b,
                x0=x0,
                rtol=rtol,
                maxiter=self.liniter,
                atol=0,
                callback=self.work_counters[self.solver_type],
                **options,
            )[0]
        else:
            raise ValueError(
                f'solver type "{self.solver_type}" not known in generic advection-diffusion implementation!'
            )
        sol[:] = x.reshape(self.nvars)
        return sol

    def _solve_fft(self, b, factor):
        """
        ``ifft(fft(b) / (1 - factor * eigenvalues))`` at the solve precision, at the state's precision.

        The right-hand side is scaled to a maximum of one first and the result scaled back, which is
        exact for a linear solve and keeps a half-precision solve clear of underflow: a correction of
        1e-10 is below ``float16``'s smallest subnormal.
        """
        scale = float(abs(b).max())
        if scale == 0.0:
            return self.xp.zeros_like(b)
        inverse = 1.0 / (1.0 - factor * self._eigenvalues)
        if self._half:
            x = self._solve_fft_half(b / scale, inverse)
        else:
            precision = np.promote_types(self._solve_dtype, np.complex64)
            x = self._fft.ifftn(self._fft.fftn((b / scale).astype(precision)) * inverse.astype(precision))
        # widen first, scale second: scaling at half precision would underflow on the way out
        x = x.astype(np.promote_types(self.dtype, np.complex64))
        x = x if self.dtype.kind == 'c' else x.real
        return x * scale

    def _solve_fft_half(self, y, inverse):
        """
        ``ifft(fft(y) * inverse)`` in half precision for ``|y| <= 1``, returned in single precision.

        Both transforms are scaled by ``1 / sqrt(N)``, so the forward one is bounded by ``sqrt(N)``
        and the inverse one by ``sqrt(N)`` times the solution: 1024 on a 1024 x 1024 grid, where an
        unscaled transform would reach 1e6 and ``float16`` stops at 65504.
        """
        unit = 1.0 / np.sqrt(y.size)
        if self.xp is np:
            # an emulation: rounded through float16 at every stage, computed in single precision

            def rounded(z):
                return z.real.astype(np.float16).astype(np.float32) + 1j * z.imag.astype(np.float16).astype(np.float32)

            spectrum = rounded(self._fft.fftn(rounded((y * unit).astype(np.complex64))))
            spectrum = rounded(spectrum * rounded(inverse.astype(np.complex64)))
            return rounded(self._fft.ifftn(spectrum, norm='forward')) * unit

        import cupy as cp
        from cupy.cuda import cufft

        shape, n = tuple(self.nvars), y.size
        if self._half_plan is None:
            # complex32 in and out, stored as float16 (re, im) pairs along the last axis
            self._half_plan = cufft.XtPlanNd(
                shape, shape, 1, n, 'E', shape, 1, n, 'E', 1, 'E', order='C', last_axis=-1, last_size=None
            )
        data = cp.zeros(shape[:-1] + (2 * shape[-1],), dtype=cp.float16)
        data[..., 0::2] = (y * unit).real
        if y.dtype.kind == 'c':
            data[..., 1::2] = (y * unit).imag
        spectrum = cp.empty_like(data)
        self._half_plan.fft(data, spectrum, cufft.CUFFT_FORWARD)
        re, im = spectrum[..., 0::2].copy(), spectrum[..., 1::2].copy()
        ir, ii = inverse.real.astype(cp.float16), inverse.imag.astype(cp.float16)
        spectrum[..., 0::2] = re * ir - im * ii
        spectrum[..., 1::2] = re * ii + im * ir
        self._half_plan.fft(spectrum, data, cufft.CUFFT_INVERSE)
        return (data[..., 0::2].astype(cp.float32) + 1j * data[..., 1::2].astype(cp.float32)) * unit
