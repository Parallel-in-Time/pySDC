import numpy as np
from mpi4py import MPI
from mpi4py_fft import PFFT

from pySDC.core.Errors import ProblemError
from pySDC.core.Problem import ptype, WorkCounter
from pySDC.implementations.datatype_classes.mesh import mesh, imex_mesh

from mpi4py_fft import newDistArray


class IMEX_Laplacian_MPIFFT(ptype):
    r"""
    Generic base class for IMEX problems using a spectral method to solve the Laplacian implicitly and a possible rest
    explicitly. The FFTs are done with``mpi4py-fft`` [1]_.

    Parameters
    ----------
    nvars : tuple, optional
        Spatial resolution
    spectral : bool, optional
        If True, the solution is computed in spectral space.
    L : float, optional
        Denotes the period of the function to be approximated for the Fourier transform.
    alpha : float, optional
        Multiplicative factor before the Laplacian
    comm : MPI.COMM_World
        Communicator for parallelisation.

    Attributes
    ----------
    fft : PFFT
        Object for parallel FFT transforms.
    X : mesh-grid
        Grid coordinates in real space.
    K2 : matrix
        Laplace operator in spectral space.

    References
    ----------
    .. [1] Lisandro Dalcin, Mikael Mortensen, David E. Keyes. Fast parallel multidimensional FFT using advanced MPI.
        Journal of Parallel and Distributed Computing (2019).
    """

    dtype_u = mesh
    dtype_f = imex_mesh

    def __init__(self, nvars=None, spectral=False, L=2 * np.pi, alpha=1.0, comm=MPI.COMM_WORLD, dtype='d'):
        """Initialization routine"""

        if nvars is None:
            nvars = (128, 128)

        if not (isinstance(nvars, tuple) and len(nvars) > 1):
            raise ProblemError('Need at least two dimensions for distributed FFTs')

        # Creating FFT structure
        self.ndim = len(nvars)
        axes = tuple(range(self.ndim))
        self.fft = PFFT(comm, list(nvars), axes=axes, dtype=dtype, collapse=True)

        # get test data to figure out type and dimensions
        tmp_u = newDistArray(self.fft, spectral)

        L = np.array([L] * self.ndim, dtype=float)

        # invoke super init, passing the communicator and the local dimensions as init
        super().__init__(init=(tmp_u.shape, comm, tmp_u.dtype))
        self._makeAttributeAndRegister('nvars', 'spectral', 'L', 'alpha', 'comm', localVars=locals(), readOnly=True)

        # get local mesh
        X = np.ogrid[self.fft.local_slice(False)]
        N = self.fft.global_shape()
        for i in range(len(N)):
            X[i] = X[i] * self.L[i] / N[i]
        self.X = [np.broadcast_to(x, self.fft.shape(False)) for x in X]

        # get local wavenumbers and Laplace operator
        s = self.fft.local_slice()
        N = self.fft.global_shape()
        k = [np.fft.fftfreq(n, 1.0 / n).astype(int) for n in N]
        K = [ki[si] for ki, si in zip(k, s)]
        Ks = np.meshgrid(*K, indexing='ij', sparse=True)
        Lp = 2 * np.pi / self.L
        for i in range(self.ndim):
            Ks[i] = (Ks[i] * Lp[i]).astype(float)
        K = [np.broadcast_to(k, self.fft.shape(True)) for k in Ks]
        K = np.array(K).astype(float)
        self.K2 = np.sum(K * K, 0, dtype=float)  # Laplacian in spectral space

        # Need this for diagnostics
        self.dx = self.L / nvars[0]
        self.dy = self.L / nvars[1]

        # work counters
        self.work_counters['rhs'] = WorkCounter()

    def eval_f(self, u, t):
        """
        Routine to evaluate the right-hand side of the problem.

        Parameters
        ----------
        u : dtype_u
            Current values of the numerical solution.
        t : float
            Current time at which the numerical solution is computed.

        Returns
        -------
        f : dtype_f
            The right-hand side of the problem.
        """

        f = self.dtype_f(self.init)

        f.impl[:] = self._eval_Laplacian(u, f.impl)

        if self.spectral:
            tmp = self.fft.backward(u)
            tmpf = self._eval_explicit_part(u, tmp)
            f.expl[:] = self.fft.forward(tmpf)

        else:
            f.expl[:] = self._eval_explicit_part(u, t, f.expl)

        self.work_counters['rhs']()
        return f

    def _eval_Laplacian(self, u, f_impl):
        if self.spectral:
            f_impl[:] = -self.alpha * self.K2 * u
        else:
            u_hat = self.fft.forward(u)
            lap_u_hat = -self.alpha * self.K2 * u_hat
            f_impl[:] = self.fft.backward(lap_u_hat, f_impl)
        return f_impl

    def _eval_explicit_part(self, u, t, f_expl):
        return f_expl

    def solve_system(self, rhs, factor, u0, t):
        """
        Simple FFT solver for the diffusion part.

        Parameters
        ----------
        rhs : dtype_f
            Right-hand side for the linear system.
        factor : float
            Abbrev. for the node-to-node stepsize (or any other factor required).
        u0 : dtype_u
            Initial guess for the iterative solver (not used here so far).
        t : float
            Current time (e.g. for time-dependent BCs).

        Returns
        -------
        me : dtype_u
            The solution as mesh.
        """
        me = self.dtype_u(self.init)
        me[:] = self._invert_Laplacian(me, factor, rhs)

        return me

    def _invert_Laplacian(self, me, factor, rhs):
        if self.spectral:
            me[:] = rhs / (1.0 + factor * self.alpha * self.K2)

        else:
            rhs_hat = self.fft.forward(rhs)
            rhs_hat /= 1.0 + factor * self.alpha * self.K2
            me[:] = self.fft.backward(rhs_hat)
        return me
