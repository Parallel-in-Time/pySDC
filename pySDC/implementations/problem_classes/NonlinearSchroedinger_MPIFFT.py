import numpy as np
from scipy.optimize import newton_krylov, root
from scipy.optimize.nonlin import NoConvergence
import scipy.sparse as sp
from mpi4py import MPI
from mpi4py_fft import PFFT

from pySDC.core.Errors import ProblemError
from pySDC.core.Problem import ptype, WorkCounter
from pySDC.implementations.datatype_classes.mesh import mesh, imex_mesh

from mpi4py_fft import newDistArray


class nonlinearschroedinger_imex(ptype):
    """
    Example implementing the nonlinear Schrödinger equation in 2-3D using mpi4py-fft for solving linear parts,
    IMEX time-stepping

    mpi4py-fft: https://mpi4py-fft.readthedocs.io/en/latest/

    Attributes:
        fft: fft object
        X: grid coordinates in real space
        K2: Laplace operator in spectral space
    """

    dtype_u = mesh
    dtype_f = imex_mesh

    def __init__(self, nvars=None, spectral=None, L=2 * np.pi, c=1.0, comm=MPI.COMM_WORLD):
        """Initialization routine"""

        if nvars is None:
            nvars = (128, 128)

        if not L == 2.0 * np.pi:
            raise ProblemError(f'Setup not implemented, L has to be 2pi, got {L}')

        if not (c == 0.0 or c == 1.0):
            raise ProblemError(f'Setup not implemented, c has to be 0 or 1, got {c}')

        if not (isinstance(nvars, tuple) and len(nvars) > 1):
            raise ProblemError('Need at least two dimensions')

        # Creating FFT structure
        self.ndim = len(nvars)
        axes = tuple(range(self.ndim))
        self.fft = PFFT(comm, list(nvars), axes=axes, dtype=np.complex128, collapse=True)

        # get test data to figure out type and dimensions
        tmp_u = newDistArray(self.fft, spectral)

        L = np.array([L] * self.ndim, dtype=float)

        # invoke super init, passing the communicator and the local dimensions as init
        super(nonlinearschroedinger_imex, self).__init__(init=(tmp_u.shape, comm, tmp_u.dtype))
        self._makeAttributeAndRegister('nvars', 'spectral', 'L', 'c', 'comm', localVars=locals(), readOnly=True)

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
        self.K2 = np.sum(K * K, 0, dtype=float)

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

        if self.spectral:
            f.impl = -self.K2 * 1j * u
            tmp = self.fft.backward(u)
            tmpf = self.ndim * self.c * 2j * np.absolute(tmp) ** 2 * tmp
            f.expl[:] = self.fft.forward(tmpf)

        else:
            u_hat = self.fft.forward(u)
            lap_u_hat = -self.K2 * 1j * u_hat
            f.impl[:] = self.fft.backward(lap_u_hat, f.impl)
            f.expl = self.ndim * self.c * 2j * np.absolute(u) ** 2 * u

        self.work_counters['rhs']()
        return f

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

        if self.spectral:
            me = rhs / (1.0 + factor * self.K2 * 1j)

        else:
            me = self.dtype_u(self.init)
            rhs_hat = self.fft.forward(rhs)
            rhs_hat /= 1.0 + factor * self.K2 * 1j
            me[:] = self.fft.backward(rhs_hat)

        return me

    def u_exact(self, t, **kwargs):
        """
        Routine to compute the exact solution at time t, see (1.3) https://arxiv.org/pdf/nlin/0702010.pdf for details

        Parameters
        ----------
        t : float
            Time of the exact solution.

        Returns
        -------
        u : dtype_u
            The exact solution.
        """
        if 'u_init' in kwargs.keys() or 't_init' in kwargs.keys():
            self.logger.warning(
                f'{type(self).__name__} uses an analytic exact solution from t=0. If you try to compute the local error, you will get the global error instead!'
            )

        def nls_exact_1D(t, x, c):
            ae = 1.0 / np.sqrt(2.0) * np.exp(1j * t)
            if c != 0:
                u = ae * ((np.cosh(t) + 1j * np.sinh(t)) / (np.cosh(t) - 1.0 / np.sqrt(2.0) * np.cos(x)) - 1.0)
            else:
                u = np.sin(x) * np.exp(-t * 1j)

            return u

        me = self.dtype_u(self.init, val=0.0)

        if self.spectral:
            tmp = nls_exact_1D(self.ndim * t, sum(self.X), self.c)
            me[:] = self.fft.forward(tmp)
        else:
            me[:] = nls_exact_1D(self.ndim * t, sum(self.X), self.c)

        return me


class nonlinearschroedinger_fully_implicit(nonlinearschroedinger_imex):
    dtype_u = mesh
    dtype_f = mesh

    def __init__(self, lintol=1e-9, liniter=99, **kwargs):
        super().__init__(**kwargs)
        self._makeAttributeAndRegister('liniter', 'lintol', localVars=locals(), readOnly=False)

        self.work_counters['newton'] = WorkCounter()

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

        if self.spectral:
            tmp = self.fft.backward(u)
            tmpf = self.ndim * self.c * 2j * np.absolute(tmp) ** 2 * tmp
            f[:] = -self.K2 * 1j * u + self.fft.forward(tmpf)

        else:
            u_hat = self.fft.forward(u)
            lap_u_hat = -self.K2 * 1j * u_hat
            f[:] = self.fft.backward(lap_u_hat) + self.ndim * self.c * 2j * np.absolute(u) ** 2 * u

        self.work_counters['rhs']()
        return f

    def solve_system(self, rhs, factor, u0, t):
        """
        Solve the nonlinear system `(1 - factor * f)(u) = rhs` using a scipy Newton-Krylov solver.
        See this page for details on the solver: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.newton_krylov.html

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

        # assemble the nonlinear function F for the solver
        def F(x):
            """
            Nonlinear function for the scipy solver.

            Args:
                x : dtype_u
                    Current solution
            """
            self.work_counters['rhs'].decrement()
            return x - factor * self.eval_f(u=x.reshape(self.init[0]), t=t).reshape(x.shape) - rhs.reshape(x.shape)

        try:
            sol = newton_krylov(
                F=F,
                xin=u0.copy(),
                maxiter=self.liniter,
                x_tol=self.lintol,
                callback=self.work_counters['newton'],
                method='gmres',
            )
        except NoConvergence as e:
            sol = e.args[0]

        me[:] = sol
        return me
