import numpy as np
from mpi4py import MPI
from mpi4py_fft import PFFT

from pySDC.core.Errors import ParameterError, ProblemError
from pySDC.core.Problem import ptype
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

    def __init__(self, problem_params, dtype_u=mesh, dtype_f=imex_mesh):
        """
        Initialization routine

        Args:
            problem_params (dict): custom parameters for the example
            dtype_u: fft data type (will be passed to parent class)
            dtype_f: fft data type wuth implicit and explicit parts (will be passed to parent class)
        """

        if 'L' not in problem_params:
            problem_params['L'] = 2.0 * np.pi
        # if 'init_type' not in problem_params:
        #     problem_params['init_type'] = 'circle'
        if 'comm' not in problem_params:
            problem_params['comm'] = MPI.COMM_WORLD
        if 'c' not in problem_params:
            problem_params['c'] = 1.0

        if not problem_params['L'] == 2.0 * np.pi:
            raise ProblemError(f'Setup not implemented, L has to be 2pi, got {problem_params["L"]}')

        if not (problem_params['c'] == 0.0 or problem_params['c'] == 1.0):
            raise ProblemError(f'Setup not implemented, c has to be 0 or 1, got {problem_params["c"]}')

        # these parameters will be used later, so assert their existence
        essential_keys = ['nvars', 'c', 'L', 'spectral']
        for key in essential_keys:
            if key not in problem_params:
                msg = 'need %s to instantiate problem, only got %s' % (key, str(problem_params.keys()))
                raise ParameterError(msg)

        if not (isinstance(problem_params['nvars'], tuple) and len(problem_params['nvars']) > 1):
            raise ProblemError('Need at least two dimensions')

        # Creating FFT structure
        self.ndim = len(problem_params['nvars'])
        axes = tuple(range(self.ndim))
        self.fft = PFFT(
            problem_params['comm'], list(problem_params['nvars']), axes=axes, dtype=np.complex128, collapse=True
        )

        # get test data to figure out type and dimensions
        tmp_u = newDistArray(self.fft, problem_params['spectral'])

        # invoke super init, passing the communicator and the local dimensions as init
        super(nonlinearschroedinger_imex, self).__init__(
            init=(tmp_u.shape, problem_params['comm'], tmp_u.dtype),
            dtype_u=dtype_u,
            dtype_f=dtype_f,
            params=problem_params,
        )

        self.L = np.array([self.params.L] * self.ndim, dtype=float)

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
        self.dx = self.params.L / problem_params['nvars'][0]
        self.dy = self.params.L / problem_params['nvars'][1]

    def eval_f(self, u, t):
        """
        Routine to evaluate the RHS

        Args:
            u (dtype_u): current values
            t (float): current time

        Returns:
            dtype_f: the RHS
        """

        f = self.dtype_f(self.init)

        if self.params.spectral:
            f.impl = -self.K2 * 1j * u
            tmp = self.fft.backward(u)
            tmpf = self.ndim * self.params.c * 2j * np.absolute(tmp) ** 2 * tmp
            f.expl[:] = self.fft.forward(tmpf)

        else:
            u_hat = self.fft.forward(u)
            lap_u_hat = -self.K2 * 1j * u_hat
            f.impl[:] = self.fft.backward(lap_u_hat, f.impl)
            f.expl = self.ndim * self.params.c * 2j * np.absolute(u) ** 2 * u

        return f

    def solve_system(self, rhs, factor, u0, t):
        """
        Simple FFT solver for the diffusion part

        Args:
            rhs (dtype_f): right-hand side for the linear system
            factor (float) : abbrev. for the node-to-node stepsize (or any other factor required)
            u0 (dtype_u): initial guess for the iterative solver (not used here so far)
            t (float): current time (e.g. for time-dependent BCs)

        Returns:
            dtype_u: solution as mesh
        """

        if self.params.spectral:
            me = rhs / (1.0 + factor * self.K2 * 1j)

        else:
            me = self.dtype_u(self.init)
            rhs_hat = self.fft.forward(rhs)
            rhs_hat /= 1.0 + factor * self.K2 * 1j
            me[:] = self.fft.backward(rhs_hat)

        return me

    def u_exact(self, t):
        """
        Routine to compute the exact solution at time t, see (1.3) https://arxiv.org/pdf/nlin/0702010.pdf for details

        Args:
            t (float): current time

        Returns:
            dtype_u: exact solution
        """

        def nls_exact_1D(t, x, c):
            ae = 1.0 / np.sqrt(2.0) * np.exp(1j * t)
            if c != 0:
                u = ae * ((np.cosh(t) + 1j * np.sinh(t)) / (np.cosh(t) - 1.0 / np.sqrt(2.0) * np.cos(x)) - 1.0)
            else:
                u = np.sin(x) * np.exp(-t * 1j)

            return u

        me = self.dtype_u(self.init, val=0.0)

        if self.params.spectral:
            tmp = nls_exact_1D(self.ndim * t, sum(self.X), self.params.c)
            me[:] = self.fft.forward(tmp)
        else:
            me[:] = nls_exact_1D(self.ndim * t, sum(self.X), self.params.c)

        return me
