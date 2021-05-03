import numpy as np
import scipy.sparse as sp
from mpi4py import MPI
from mpi4py_fft import PFFT

from pySDC.core.Errors import ParameterError, ProblemError
from pySDC.core.Problem import ptype
from pySDC.implementations.datatype_classes.mesh import mesh, imex_mesh, comp2_mesh

from mpi4py_fft import newDistArray


class grayscott_imex_diffusion(ptype):
    """
    Example implementing the Gray-Scott equation in 2-3D using mpi4py-fft for solving linear parts,
    IMEX time-stepping (implicit diffusion, explicit reaction)

    mpi4py-fft: https://mpi4py-fft.readthedocs.io/en/latest/

    Attributes:
        fft: fft object
        X: grid coordinates in real space
        ndim: number of spatial dimensions
        Ku: Laplace operator in spectral space (u component)
        Kv: Laplace operator in spectral space (v component)
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
            problem_params['L'] = 2.0
        # if 'init_type' not in problem_params:
        #     problem_params['init_type'] = 'circle'
        if 'comm' not in problem_params:
            problem_params['comm'] = MPI.COMM_WORLD

        # these parameters will be used later, so assert their existence
        essential_keys = ['nvars', 'Du', 'Dv', 'A', 'B', 'spectral']
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
                        collapse=True, backend='fftw')

        # get test data to figure out type and dimensions
        tmp_u = newDistArray(self.fft, problem_params['spectral'])

        # add two components to contain field and temperature
        self.ncomp = 2
        sizes = tmp_u.shape + (self.ncomp,)

        # invoke super init, passing the communicator and the local dimensions as init
        super(grayscott_imex_diffusion, self).__init__(init=(sizes, problem_params['comm'], tmp_u.dtype),
                                                       dtype_u=dtype_u, dtype_f=dtype_f, params=problem_params)

        L = np.array([self.params.L] * self.ndim, dtype=float)

        # get local mesh
        X = np.ogrid[self.fft.local_slice(False)]
        N = self.fft.global_shape()
        for i in range(len(N)):
            X[i] = -L[i] / 2 + (X[i] * L[i] / N[i])
        self.X = [np.broadcast_to(x, self.fft.shape(False)) for x in X]

        # get local wavenumbers and Laplace operator
        s = self.fft.local_slice()
        N = self.fft.global_shape()
        k = [np.fft.fftfreq(n, 1. / n).astype(int) for n in N[:-1]]
        k.append(np.fft.rfftfreq(N[-1], 1. / N[-1]).astype(int))
        K = [ki[si] for ki, si in zip(k, s)]
        Ks = np.meshgrid(*K, indexing='ij', sparse=True)
        Lp = 2 * np.pi / L
        for i in range(self.ndim):
            Ks[i] = (Ks[i] * Lp[i]).astype(float)
        K = [np.broadcast_to(k, self.fft.shape(True)) for k in Ks]
        K = np.array(K).astype(float)
        self.K2 = np.sum(K * K, 0, dtype=float)
        self.Ku = -self.K2 * self.params.Du
        self.Kv = -self.K2 * self.params.Dv

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

            f.impl[..., 0] = self.Ku * u[..., 0]
            f.impl[..., 1] = self.Kv * u[..., 1]
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
            lap_u_hat = self.Ku * u_hat
            f.impl[..., 0] = self.fft.backward(lap_u_hat, f.impl[..., 0])
            u_hat = self.fft.forward(u[..., 1])
            lap_u_hat = self.Kv * u_hat
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

            me[..., 0] = rhs[..., 0] / (1.0 - factor * self.Ku)
            me[..., 1] = rhs[..., 1] / (1.0 - factor * self.Kv)

        else:

            rhs_hat = self.fft.forward(rhs[..., 0])
            rhs_hat /= (1.0 - factor * self.Ku)
            me[..., 0] = self.fft.backward(rhs_hat, me[..., 0])
            rhs_hat = self.fft.forward(rhs[..., 1])
            rhs_hat /= (1.0 - factor * self.Kv)
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

        # tmpu = np.load('data/u_0001.npy')
        # tmpv = np.load('data/v_0001.npy')
        #
        # me[..., 0] = self.fft.forward(tmpu)
        # me[..., 1] = self.fft.forward(tmpv)

        return me


class grayscott_imex_linear(grayscott_imex_diffusion):

    def __init__(self, problem_params, dtype_u=mesh, dtype_f=imex_mesh):
        """
        Init routine for the IMEX problem class with linear splitting
        """

        super(grayscott_imex_linear, self).__init__(problem_params, dtype_u=dtype_u, dtype_f=dtype_f)

        self.Ku -= self.params.A
        self.Kv -= self.params.B

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

            f.impl[..., 0] = self.Ku * u[..., 0]
            f.impl[..., 1] = self.Kv * u[..., 1]
            tmpu = newDistArray(self.fft, False)
            tmpv = newDistArray(self.fft, False)
            tmpu[:] = self.fft.backward(u[..., 0], tmpu)
            tmpv[:] = self.fft.backward(u[..., 1], tmpv)
            tmpfu = -tmpu * tmpv ** 2 + self.params.A
            tmpfv = tmpu * tmpv ** 2
            f.expl[..., 0] = self.fft.forward(tmpfu)
            f.expl[..., 1] = self.fft.forward(tmpfv)

        else:

            u_hat = self.fft.forward(u[..., 0])
            lap_u_hat = self.Ku * u_hat
            f.impl[..., 0] = self.fft.backward(lap_u_hat, f.impl[..., 0])
            u_hat = self.fft.forward(u[..., 1])
            lap_u_hat = self.Kv * u_hat
            f.impl[..., 1] = self.fft.backward(lap_u_hat, f.impl[..., 1])
            f.expl[..., 0] = -u[..., 0] * u[..., 1] ** 2 + self.params.A
            f.expl[..., 1] = u[..., 0] * u[..., 1] ** 2

        return f


class grayscott_mi_diffusion(grayscott_imex_diffusion):

    def __init__(self, problem_params, dtype_u=mesh, dtype_f=comp2_mesh):
        """
        Init routine for the multi-implicit problem class with diffusion splitting
        """

        if 'newton_maxiter' not in problem_params:
            raise ParameterError('need newton_maxiter as parameter for the problem class')
        if 'newton_tol' not in problem_params:
            raise ParameterError('need newton_tol as parameter for the problem class')

        super(grayscott_mi_diffusion, self).__init__(problem_params, dtype_u=dtype_u, dtype_f=dtype_f)

        # This may not run in parallel yet..
        assert self.params.comm.Get_size() == 1

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

            f.comp1[..., 0] = self.Ku * u[..., 0]
            f.comp1[..., 1] = self.Kv * u[..., 1]
            tmpu = newDistArray(self.fft, False)
            tmpv = newDistArray(self.fft, False)
            tmpu[:] = self.fft.backward(u[..., 0], tmpu)
            tmpv[:] = self.fft.backward(u[..., 1], tmpv)
            tmpfu = -tmpu * tmpv ** 2 + self.params.A * (1 - tmpu)
            tmpfv = tmpu * tmpv ** 2 - self.params.B * tmpv
            f.comp2[..., 0] = self.fft.forward(tmpfu)
            f.comp2[..., 1] = self.fft.forward(tmpfv)

        else:

            u_hat = self.fft.forward(u[..., 0])
            lap_u_hat = self.Ku * u_hat
            f.comp1[..., 0] = self.fft.backward(lap_u_hat, f.comp1[..., 0])
            u_hat = self.fft.forward(u[..., 1])
            lap_u_hat = self.Kv * u_hat
            f.comp1[..., 1] = self.fft.backward(lap_u_hat, f.comp1[..., 1])
            f.comp2[..., 0] = -u[..., 0] * u[..., 1] ** 2 + self.params.A * (1 - u[..., 0])
            f.comp2[..., 1] = u[..., 0] * u[..., 1] ** 2 - self.params.B * u[..., 1]

        return f

    def solve_system_1(self, rhs, factor, u0, t):
        """
        Solver for the first component, can just call the super function

        Args:
            rhs (dtype_f): right-hand side for the linear system
            factor (float) : abbrev. for the node-to-node stepsize (or any other factor required)
            u0 (dtype_u): initial guess for the iterative solver (not used here so far)
            t (float): current time (e.g. for time-dependent BCs)

        Returns:
            dtype_u: solution as mesh
        """

        me = super(grayscott_mi_diffusion, self).solve_system(rhs, factor, u0, t)
        return me

    def solve_system_2(self, rhs, factor, u0, t):
        """
        Newton-Solver for the second component

        Args:
            rhs (dtype_f): right-hand side for the linear system
            factor (float) : abbrev. for the node-to-node stepsize (or any other factor required)
            u0 (dtype_u): initial guess for the iterative solver (not used here so far)
            t (float): current time (e.g. for time-dependent BCs)

        Returns:
            dtype_u: solution as mesh
        """
        u = self.dtype_u(u0)

        if self.params.spectral:
            tmpu = newDistArray(self.fft, False)
            tmpv = newDistArray(self.fft, False)
            tmpu[:] = self.fft.backward(u[..., 0], tmpu)
            tmpv[:] = self.fft.backward(u[..., 1], tmpv)
            tmprhsu = newDistArray(self.fft, False)
            tmprhsv = newDistArray(self.fft, False)
            tmprhsu[:] = self.fft.backward(rhs[..., 0], tmprhsu)
            tmprhsv[:] = self.fft.backward(rhs[..., 1], tmprhsv)

        else:
            tmpu = u[..., 0]
            tmpv = u[..., 1]
            tmprhsu = rhs[..., 0]
            tmprhsv = rhs[..., 1]

        # start newton iteration
        n = 0
        res = 99
        while n < self.params.newton_maxiter:
            # print(n, res)
            # form the function g with g(u) = 0
            tmpgu = tmpu - tmprhsu - factor * (-tmpu * tmpv ** 2 + self.params.A * (1 - tmpu))
            tmpgv = tmpv - tmprhsv - factor * (tmpu * tmpv ** 2 - self.params.B * tmpv)

            # if g is close to 0, then we are done
            res = max(np.linalg.norm(tmpgu, np.inf), np.linalg.norm(tmpgv, np.inf))
            if res < self.params.newton_tol:
                break

            # assemble dg
            dg00 = 1 - factor * (-tmpv ** 2 - self.params.A)
            dg01 = -factor * (-2 * tmpu * tmpv)
            dg10 = -factor * (tmpv ** 2)
            dg11 = 1 - factor * (2 * tmpu * tmpv - self.params.B)

            # interleave and unravel to put into sparse matrix
            dg00I = np.ravel(np.kron(dg00, np.array([1, 0])))
            dg01I = np.ravel(np.kron(dg01, np.array([1, 0])))
            dg10I = np.ravel(np.kron(dg10, np.array([1, 0])))
            dg11I = np.ravel(np.kron(dg11, np.array([0, 1])))

            # put into sparse matrix
            dg = sp.diags(dg00I, offsets=0) + sp.diags(dg11I, offsets=0)
            dg += sp.diags(dg01I, offsets=1, shape=dg.shape) + sp.diags(dg10I, offsets=-1, shape=dg.shape)

            # interleave g terms to apply inverse to it
            g = np.kron(tmpgu.flatten(), np.array([1, 0])) + np.kron(tmpgv.flatten(), np.array([0, 1]))
            # invert dg matrix
            b = sp.linalg.spsolve(dg, g)
            # update real space vectors
            tmpu[:] -= b[::2].reshape(self.params.nvars)
            tmpv[:] -= b[1::2].reshape(self.params.nvars)

            # increase iteration count
            n += 1

        if np.isnan(res) and self.params.stop_at_nan:
            raise ProblemError('Newton got nan after %i iterations, aborting...' % n)
        elif np.isnan(res):
            self.logger.warning('Newton got nan after %i iterations...' % n)

        if n == self.params.newton_maxiter:
            self.logger.warning('Newton did not converge after %i iterations, error is %s' % (n, res))

        # self.newton_ncalls += 1
        # self.newton_itercount += n
        me = self.dtype_u(self.init)
        if self.params.spectral:
            me[..., 0] = self.fft.forward(tmpu)
            me[..., 1] = self.fft.forward(tmpv)
        else:
            me[..., 0] = tmpu
            me[..., 1] = tmpv
        return me


class grayscott_mi_linear(grayscott_imex_linear):

    def __init__(self, problem_params, dtype_u=mesh, dtype_f=comp2_mesh):
        """
        Init routine for the multi-implicit problem class with linear splitting
        """

        if 'newton_maxiter' not in problem_params:
            raise ParameterError('need newton_maxiter as parameter for the problem class')
        if 'newton_tol' not in problem_params:
            raise ParameterError('need newton_tol as parameter for the problem class')

        super(grayscott_mi_linear, self).__init__(problem_params, dtype_u=dtype_u, dtype_f=dtype_f)

        # This may not run in parallel yet..
        assert self.params.comm.Get_size() == 1

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

            f.comp1[..., 0] = self.Ku * u[..., 0]
            f.comp1[..., 1] = self.Kv * u[..., 1]
            tmpu = newDistArray(self.fft, False)
            tmpv = newDistArray(self.fft, False)
            tmpu[:] = self.fft.backward(u[..., 0], tmpu)
            tmpv[:] = self.fft.backward(u[..., 1], tmpv)
            tmpfu = -tmpu * tmpv ** 2 + self.params.A
            tmpfv = tmpu * tmpv ** 2
            f.comp2[..., 0] = self.fft.forward(tmpfu)
            f.comp2[..., 1] = self.fft.forward(tmpfv)

        else:

            u_hat = self.fft.forward(u[..., 0])
            lap_u_hat = self.Ku * u_hat
            f.comp1[..., 0] = self.fft.backward(lap_u_hat, f.comp1[..., 0])
            u_hat = self.fft.forward(u[..., 1])
            lap_u_hat = self.Kv * u_hat
            f.comp1[..., 1] = self.fft.backward(lap_u_hat, f.comp1[..., 1])
            f.comp2[..., 0] = -u[..., 0] * u[..., 1] ** 2 + self.params.A
            f.comp2[..., 1] = u[..., 0] * u[..., 1] ** 2

        return f

    def solve_system_1(self, rhs, factor, u0, t):
        """
        Solver for the first component, can just call the super function

        Args:
            rhs (dtype_f): right-hand side for the linear system
            factor (float) : abbrev. for the node-to-node stepsize (or any other factor required)
            u0 (dtype_u): initial guess for the iterative solver (not used here so far)
            t (float): current time (e.g. for time-dependent BCs)

        Returns:
            dtype_u: solution as mesh
        """

        me = super(grayscott_mi_linear, self).solve_system(rhs, factor, u0, t)
        return me

    def solve_system_2(self, rhs, factor, u0, t):
        """
        Newton-Solver for the second component

        Args:
            rhs (dtype_f): right-hand side for the linear system
            factor (float) : abbrev. for the node-to-node stepsize (or any other factor required)
            u0 (dtype_u): initial guess for the iterative solver (not used here so far)
            t (float): current time (e.g. for time-dependent BCs)

        Returns:
            dtype_u: solution as mesh
        """
        u = self.dtype_u(u0)

        if self.params.spectral:
            tmpu = newDistArray(self.fft, False)
            tmpv = newDistArray(self.fft, False)
            tmpu[:] = self.fft.backward(u[..., 0], tmpu)
            tmpv[:] = self.fft.backward(u[..., 1], tmpv)
            tmprhsu = newDistArray(self.fft, False)
            tmprhsv = newDistArray(self.fft, False)
            tmprhsu[:] = self.fft.backward(rhs[..., 0], tmprhsu)
            tmprhsv[:] = self.fft.backward(rhs[..., 1], tmprhsv)

        else:
            tmpu = u[..., 0]
            tmpv = u[..., 1]
            tmprhsu = rhs[..., 0]
            tmprhsv = rhs[..., 1]

        # start newton iteration
        n = 0
        res = 99
        while n < self.params.newton_maxiter:
            # print(n, res)
            # form the function g with g(u) = 0
            tmpgu = tmpu - tmprhsu - factor * (-tmpu * tmpv ** 2 + self.params.A)
            tmpgv = tmpv - tmprhsv - factor * (tmpu * tmpv ** 2)

            # if g is close to 0, then we are done
            res = max(np.linalg.norm(tmpgu, np.inf), np.linalg.norm(tmpgv, np.inf))
            if res < self.params.newton_tol:
                break

            # assemble dg
            dg00 = 1 - factor * (-tmpv ** 2)
            dg01 = -factor * (-2 * tmpu * tmpv)
            dg10 = -factor * (tmpv ** 2)
            dg11 = 1 - factor * (2 * tmpu * tmpv)

            # interleave and unravel to put into sparse matrix
            dg00I = np.ravel(np.kron(dg00, np.array([1, 0])))
            dg01I = np.ravel(np.kron(dg01, np.array([1, 0])))
            dg10I = np.ravel(np.kron(dg10, np.array([1, 0])))
            dg11I = np.ravel(np.kron(dg11, np.array([0, 1])))

            # put into sparse matrix
            dg = sp.diags(dg00I, offsets=0) + sp.diags(dg11I, offsets=0)
            dg += sp.diags(dg01I, offsets=1, shape=dg.shape) + sp.diags(dg10I, offsets=-1, shape=dg.shape)

            # interleave g terms to apply inverse to it
            g = np.kron(tmpgu.flatten(), np.array([1, 0])) + np.kron(tmpgv.flatten(), np.array([0, 1]))
            # invert dg matrix
            b = sp.linalg.spsolve(dg, g)
            # update real-space vectors
            tmpu[:] -= b[::2].reshape(self.params.nvars)
            tmpv[:] -= b[1::2].reshape(self.params.nvars)

            # increase iteration count
            n += 1

        if np.isnan(res) and self.params.stop_at_nan:
            raise ProblemError('Newton got nan after %i iterations, aborting...' % n)
        elif np.isnan(res):
            self.logger.warning('Newton got nan after %i iterations...' % n)

        if n == self.params.newton_maxiter:
            self.logger.warning('Newton did not converge after %i iterations, error is %s' % (n, res))

        # self.newton_ncalls += 1
        # self.newton_itercount += n
        me = self.dtype_u(self.init)
        if self.params.spectral:
            me[..., 0] = self.fft.forward(tmpu)
            me[..., 1] = self.fft.forward(tmpv)
        else:
            me[..., 0] = tmpu
            me[..., 1] = tmpv
        return me
