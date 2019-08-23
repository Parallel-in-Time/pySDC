import numpy as np
from mpi4py import MPI
from mpi4py_fft import PFFT

from pySDC.core.Errors import ParameterError, ProblemError
from pySDC.core.Problem import ptype
from pySDC.implementations.datatype_classes.parallel_mesh import parallel_mesh, parallel_imex_mesh

from mpi4py_fft import newDistArray


class grayscott_imex(ptype):
    """
    Example implementing the Gray-Scott equation in 2-3D using mpi4py-fft for solving linear parts,
    IMEX time-stepping

    mpi4py-fft: https://mpi4py-fft.readthedocs.io/en/latest/

    Attributes:
        fft: fft object
        X: grid coordinates in real space
        K2: Laplace operator in spectral space
    """

    def __init__(self, problem_params, dtype_u=parallel_mesh, dtype_f=parallel_imex_mesh):
        """
        Initialization routine

        Args:
            problem_params (dict): custom parameters for the example
            dtype_u: fft data type (will be passed to parent class)
            dtype_f: fft data type wuth implicit and explicit parts (will be passed to parent class)
        """

        if 'L' not in problem_params:
            problem_params['L'] = 2.0
        # if 'init_type' not in problem_params:
        #     problem_params['init_type'] = 'circle'
        if 'comm' not in problem_params:
            problem_params['comm'] = MPI.COMM_WORLD

        # these parameters will be used later, so assert their existence
        essential_keys = essential_keys = ['nvars', 'Du', 'Dv', 'A', 'B', 'spectral']
        for key in essential_keys:
            if key not in problem_params:
                msg = 'need %s to instantiate problem, only got %s' % (key, str(problem_params.keys()))
                raise ParameterError(msg)

        if not (isinstance(problem_params['nvars'], tuple) and len(problem_params['nvars']) > 1):
            raise ProblemError('Need at least two dimensions')

        # Creating FFT structure
        self.ndim = len(problem_params['nvars'])
        axes = tuple(range(self.ndim))
        self.fft = PFFT(problem_params['comm'], list(problem_params['nvars']), axes=axes, dtype=np.float64,
                        collapse=True)

        # get test data to figure out type and dimensions
        tmp_u = newDistArray(self.fft, problem_params['spectral'])

        # add two components to contain field and temperature
        self.ncomp = 2
        sizes = tmp_u.shape + (self.ncomp,)

        # invoke super init, passing the communicator and the local dimensions as init
        super(grayscott_imex, self).__init__(init=(sizes, problem_params['comm'], tmp_u.dtype),
                                             dtype_u=dtype_u, dtype_f=dtype_f, params=problem_params)

        self.L = np.array([self.params.L] * self.ndim, dtype=float)

        # get local mesh
        X = np.ogrid[self.fft.local_slice(False)]
        N = self.fft.global_shape()
        for i in range(len(N)):
            X[i] = -self.L[i] / 2 + (X[i] * self.L[i] / N[i])
        self.X = [np.broadcast_to(x, self.fft.shape(False)) for x in X]

        # get local wavenumbers and Laplace operator
        s = self.fft.local_slice()
        N = self.fft.global_shape()
        k = [np.fft.fftfreq(n, 1. / n).astype(int) for n in N[:-1]]
        k.append(np.fft.rfftfreq(N[-1], 1. / N[-1]).astype(int))
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

            f.impl[..., 0] = -self.K2 * self.params.Du * u[..., 0]
            f.impl[..., 1] = -self.K2 * self.params.Dv * u[..., 1]
            tmpu = newDistArray(self.fft, False)
            tmpv = newDistArray(self.fft, False)
            tmpu[:] = self.fft.backward(u[..., 0], tmpu)
            tmpv[:] = self.fft.backward(u[..., 1], tmpv)
            tmpfu = -tmpu * tmpv ** 2 + self.params.A * (1 - tmpu)
            tmpfv = tmpu * tmpv ** 2 - self.params.B * tmpv
            f.expl[..., 0] = self.fft.forward(tmpfu)
            f.expl[..., 1] = self.fft.forward(tmpfv)

        else:

            u_hat = self.fft.forward(u[..., 0])
            lap_u_hat = -self.K2 * self.params.Du * u_hat
            f.impl[..., 0] = self.fft.backward(lap_u_hat, f.impl[..., 0])
            u_hat = self.fft.forward(u[..., 1])
            lap_u_hat = -self.K2 * self.params.Dv * u_hat
            f.impl[..., 1] = self.fft.backward(lap_u_hat, f.impl[..., 1])

            f.expl[..., 0] = -u[..., 0] * u[..., 1] ** 2 + self.params.A * (1 - u[..., 0])
            f.expl[..., 1] = u[..., 0] * u[..., 1] ** 2 - self.params.B * u[..., 1]

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

        me = self.dtype_u(self.init)
        if self.params.spectral:

            me[..., 0] = rhs[..., 0] / (1.0 + factor * self.K2 * self.params.Du)
            me[..., 1] = rhs[..., 1] / (1.0 + factor * self.K2 * self.params.Dv)

        else:

            rhs_hat = self.fft.forward(rhs[..., 0])
            rhs_hat /= (1.0 + factor * self.K2 * self.params.Du)
            me[..., 0] = self.fft.backward(rhs_hat, me[..., 0])
            rhs_hat = self.fft.forward(rhs[..., 1])
            rhs_hat /= (1.0 + factor * self.K2 * self.params.Dv)
            me[..., 1] = self.fft.backward(rhs_hat, me[..., 1])

        return me

    def u_exact(self, t):
        """
        Routine to compute the exact solution at time t=0, see https://www.chebfun.org/examples/pde/GrayScott.html

        Args:
            t (float): current time

        Returns:
            dtype_u: exact solution
        """
        assert t == 0.0, 'Exact solution only valid as initial condition'
        assert self.ndim == 2, 'The initial conditions are 2D for now..'

        me = self.dtype_u(self.init, val=0.0)

        # This assumes that the box is [-L/2, L/2]^2
        if self.params.spectral:
            tmp = 1.0 - np.exp(-80.0 * ((self.X[0] + 0.05) ** 2 + (self.X[1] + 0.02) ** 2))
            me[..., 0] = self.fft.forward(tmp)
            tmp = np.exp(-80.0 * ((self.X[0] - 0.05) ** 2 + (self.X[1] - 0.02) ** 2))
            me[..., 1] = self.fft.forward(tmp)
        else:
            me[..., 0] = 1.0 - np.exp(-80.0 * ((self.X[0] + 0.05) ** 2 + (self.X[1] + 0.02) ** 2))
            me[..., 1] = np.exp(-80.0 * ((self.X[0] - 0.05) ** 2 + (self.X[1] - 0.02) ** 2))

        return me
