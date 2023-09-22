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
    r"""
    Example implementing the :math:`N`-dimensional nonlinear Schrödinger equation with periodic boundary conditions

    .. math::
        \frac{\partial u}{\partial t} = -i \Delta u + 2 c i |u|^2 u

    for fixed parameter :math:`c` and :math:`N=2, 3`. The linear parts of the problem will be solved using
    ``mpi4py-fft`` [1]_. *Semi-explicit* time-stepping is used here to solve the problem in the temporal dimension, i.e., the
    Laplacian will be handled implicitly.

    Parameters
    ----------
    nvars : tuple, optional
        Spatial resolution
    spectral : bool, optional
        If True, the solution is computed in spectral space.
    L : float, optional
        Denotes the period of the function to be approximated for the Fourier transform.
    c : float, optional
        Nonlinearity parameter.
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

    def __init__(self, nvars=None, spectral=False, L=2 * np.pi, c=1.0, comm=MPI.COMM_WORLD):
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
        r"""
        Routine to compute the exact solution at time :math:`t`, see (1.3) https://arxiv.org/pdf/nlin/0702010.pdf for details

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
    r"""
    Example implementing the :math:`N`-dimensional nonlinear Schrödinger equation with periodic boundary conditions

    .. math::
        \frac{\partial u}{\partial t} = -i \Delta u + 2 c i |u|2 u

    for fixed parameter :math:`c` and :math:`N=2, 3`. The linear parts of the problem will be discretized using
    ``mpi4py-fft`` [1]_. For time-stepping, the problem will be solved *fully-implicitly*, i.e., the nonlinear system containing
    the full right-hand side is solved by GMRES method.

    References
    ----------
    .. [1] https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.newton_krylov.html
    """

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
        r"""
        Solve the nonlinear system :math:`(1 - factor \cdot f)(\vec{u}) = \vec{rhs}` using a ``SciPy`` Newton-Krylov
        solver. See page [1]_ for details on the solver.

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
            r"""
            Nonlinear function for the ``SciPy`` solver.

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
