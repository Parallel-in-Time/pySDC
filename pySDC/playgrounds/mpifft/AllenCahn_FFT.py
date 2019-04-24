import numpy as np
from mpi4py import MPI
from mpi4py_fft import PFFT

from pySDC.core.Errors import ParameterError, ProblemError
from pySDC.core.Problem import ptype
from pySDC.playgrounds.mpifft.FFT_datatype import fft_datatype, rhs_imex_fft

from mpi4py_fft import newDistArray


class allencahn_imex(ptype):
    """
    Example implementing Allen-Cahn equation in 2-3D using PMESH for solving linear parts, IMEX time-stepping

    PMESH: https://github.com/rainwoodman/pmesh

    Attributes:
        xvalues: grid points in space
        dx: mesh width
    """

    def __init__(self, problem_params, dtype_u=fft_datatype, dtype_f=rhs_imex_fft):
        """
        Initialization routine

        Args:
            problem_params (dict): custom parameters for the example
            dtype_u: pmesh data type (will be passed to parent class)
            dtype_f: pmesh data type wuth implicit and explicit parts (will be passed to parent class)
        """

        if 'L' not in problem_params:
            problem_params['L'] = 1.0
        if 'init_type' not in problem_params:
            problem_params['init_type'] = 'circle'
        if 'comm' not in problem_params:
            problem_params['comm'] = None
        if 'dw' not in problem_params:
            problem_params['dw'] = 0.0

        # these parameters will be used later, so assert their existence
        essential_keys = ['nvars', 'eps', 'L', 'radius', 'dw']
        for key in essential_keys:
            if key not in problem_params:
                msg = 'need %s to instantiate problem, only got %s' % (key, str(problem_params.keys()))
                raise ParameterError(msg)

        if not (isinstance(problem_params['nvars'], tuple) and len(problem_params['nvars']) > 1):
            raise ProblemError('Need at least two dimensions')

        # Creating FFT structure
        ndim = len(problem_params['nvars'])
        axes = tuple(range(ndim))
        self.fft = PFFT(problem_params['comm'], list(problem_params['nvars']), axes=axes, dtype=np.float, collapse=True)

        # invoke super init, passing the communicator and the local dimensions as init
        super(allencahn_imex, self).__init__(init=self.fft, dtype_u=dtype_u, dtype_f=dtype_f,
                                             params=problem_params)

        L = np.array([self.params.L] * ndim, dtype=float)

        # get local mesh
        X = np.ogrid[self.fft.local_slice(False)]
        N = self.fft.global_shape()
        for i in range(len(N)):
            X[i] = (X[i] * L[i] / N[i])
        self.X = [np.broadcast_to(x, self.fft.shape(False)) for x in X]

        # get local wavenumbers and Laplace operator
        s = self.fft.local_slice()
        N = self.fft.global_shape()
        k = [np.fft.fftfreq(n, 1. / n).astype(int) for n in N[:-1]]
        k.append(np.fft.rfftfreq(N[-1], 1. / N[-1]).astype(int))
        K = [ki[si] for ki, si in zip(k, s)]
        Ks = np.meshgrid(*K, indexing='ij', sparse=True)
        Lp = 2 * np.pi / L
        for i in range(ndim):
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

        u_hat = self.fft.forward(u.values)
        lap_u_hat = -self.K2 * u_hat
        f.impl.values = self.fft.backward(lap_u_hat, f.impl.values)

        if self.params.eps > 0:
            f.expl.values = - 2.0 / self.params.eps ** 2 * u.values * (1.0 - u.values) * (1.0 - 2.0 * u.values) - \
                6.0 * self.params.dw * u.values * (1.0 - u.values)

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
        rhs_hat = self.fft.forward(rhs.values)
        rhs_hat /= (1.0 + factor * self.K2)
        me.values = self.fft.backward(rhs_hat, me.values)

        return me

    def u_exact(self, t):
        """
        Routine to compute the exact solution at time t

        Args:
            t (float): current time

        Returns:
            dtype_u: exact solution
        """

        assert t == 0, 'ERROR: u_exact only valid for t=0'
        me = self.dtype_u(self.init, val=0)
        if self.params.init_type == 'circle':
            r2 = (self.X[0] - 0.5) ** 2 + (self.X[1] - 0.5) ** 2
            me.values[:] = 0.5 * (1.0 + np.tanh((self.params.radius - np.sqrt(r2)) / (np.sqrt(2) * self.params.eps)))
        elif self.params.init_type == 'circle_rand':
            ndim = len(me.values.shape)
            L = int(self.params.L)
            # get random radii for circles/spheres
            np.random.seed(1)
            lbound = 3.0 * self.params.eps
            ubound = 0.5 - self.params.eps
            rand_radii = (ubound - lbound) * np.random.random_sample(size=tuple([L] * ndim)) + lbound
            # distribute circles/spheres
            if ndim == 2:
                for i in range(0, L):
                    for j in range(0, L):
                        # build radius
                        r2 = (self.X[0] + i - L + 0.5) ** 2 + (self.X[1] + j - L + 0.5) ** 2
                        # add this blob, shifted by 1 to avoid issues with adding up negative contributions
                        me.values += np.tanh((rand_radii[i, j] - np.sqrt(r2)) / (np.sqrt(2) * self.params.eps)) + 1
            # normalize to [0,1]
            me.values *= 0.5
            assert np.all(me.values <= 1.0)
        else:
            raise NotImplementedError('type of initial value not implemented, got %s' % self.params.init_type)

        return me


class allencahn_imex_timeforcing(allencahn_imex):
    """
    Example implementing Allen-Cahn equation in 2-3D using PMESH for solving linear parts, IMEX time-stepping,
    time-dependent forcing
    """

    def eval_f(self, u, t):
        """
        Routine to evaluate the RHS

        Args:
            u (dtype_u): current values
            t (float): current time

        Returns:
            dtype_f: the RHS
        """

        def Laplacian(k, v):
            k2 = sum(ki ** 2 for ki in k)
            return -k2 * v

        f = self.dtype_f(self.init)
        tmp_u = self.pm.create(type='real', value=u.values)
        f.impl.values = tmp_u.r2c().apply(Laplacian, out=Ellipsis).c2r(out=Ellipsis).value

        if self.params.eps > 0:
            f.expl.values = - 2.0 / self.params.eps ** 2 * u.values * (1.0 - u.values) * (1.0 - 2.0 * u.values)

        # build sum over RHS without driving force
        Rt_local = f.impl.values.sum() + f.expl.values.sum()
        if self.pm.comm is not None:
            Rt_global = self.pm.comm.allreduce(sendobj=Rt_local, op=MPI.SUM)
        else:
            Rt_global = Rt_local

        # build sum over driving force term
        Ht_local = np.sum(6.0 * u.values * (1.0 - u.values))
        if self.pm.comm is not None:
            Ht_global = self.pm.comm.allreduce(sendobj=Ht_local, op=MPI.SUM)
        else:
            Ht_global = Rt_local

        # add/substract time-dependent driving force
        dw = Rt_global / Ht_global
        f.expl.values -= 6.0 * dw * u.values * (1.0 - u.values)

        return f


class allencahn_imex_stab(allencahn_imex):
    """
    Example implementing Allen-Cahn equation in 2-3D using PMESH for solving linear parts, IMEX time-stepping with
    stabilized splitting
    """

    def eval_f(self, u, t):
        """
        Routine to evaluate the RHS

        Args:
            u (dtype_u): current values
            t (float): current time

        Returns:
            dtype_f: the RHS
        """

        def Laplacian(k, v):
            k2 = sum(ki ** 2 for ki in k) + 1.0 / self.params.eps ** 2
            return -k2 * v

        f = self.dtype_f(self.init)
        tmp_u = self.pm.create(type='real', value=u.values)
        f.impl.values = tmp_u.r2c().apply(Laplacian, out=Ellipsis).c2r(out=Ellipsis).value
        if self.params.eps > 0:
            f.expl.values = - 2.0 / self.params.eps ** 2 * u.values * (1.0 - u.values) * (1.0 - 2.0 * u.values) - \
                6.0 * self.params.dw * u.values * (1.0 - u.values) + \
                1.0 / self.params.eps ** 2 * u.values
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

        def linear_solve(k, v):
            k2 = sum(ki ** 2 for ki in k) + 1.0 / self.params.eps ** 2
            return 1.0 / (1.0 + factor * k2) * v

        me = self.dtype_u(self.init)
        tmp_rhs = self.pm.create(type='real', value=rhs.values)
        me.values = tmp_rhs.r2c().apply(linear_solve, out=Ellipsis).c2r(out=Ellipsis).value

        return me
