import numpy as np
import scipy.sparse as sp
from mpi4py import MPI
from mpi4py_fft import PFFT

from pySDC.core.Errors import ProblemError
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

    dtype_u = mesh
    dtype_f = imex_mesh

    def __init__(self, nvars, Du, Dv, A, B, spectral, L=2.0, comm=MPI.COMM_WORLD):
        if not (isinstance(nvars, tuple) and len(nvars) > 1):
            raise ProblemError('Need at least two dimensions')

        # Creating FFT structure
        self.ndim = len(nvars)
        axes = tuple(range(self.ndim))
        self.fft = PFFT(
            comm,
            list(nvars),
            axes=axes,
            dtype=np.float64,
            collapse=True,
            backend='fftw',
        )

        # get test data to figure out type and dimensions
        tmp_u = newDistArray(self.fft, spectral)

        # add two components to contain field and temperature
        self.ncomp = 2
        sizes = tmp_u.shape + (self.ncomp,)

        # invoke super init, passing the communicator and the local dimensions as init
        super().__init__(init=(sizes, comm, tmp_u.dtype))
        self._makeAttributeAndRegister(
            'nvars', 'Du', 'Dv', 'A', 'B', 'spectral', 'L', 'comm', localVars=locals(), readOnly=True
        )

        L = np.array([self.L] * self.ndim, dtype=float)

        # get local mesh
        X = np.ogrid[self.fft.local_slice(False)]
        N = self.fft.global_shape()
        for i in range(len(N)):
            X[i] = -L[i] / 2 + (X[i] * L[i] / N[i])
        self.X = [np.broadcast_to(x, self.fft.shape(False)) for x in X]

        # get local wavenumbers and Laplace operator
        s = self.fft.local_slice()
        N = self.fft.global_shape()
        k = [np.fft.fftfreq(n, 1.0 / n).astype(int) for n in N[:-1]]
        k.append(np.fft.rfftfreq(N[-1], 1.0 / N[-1]).astype(int))
        K = [ki[si] for ki, si in zip(k, s)]
        Ks = np.meshgrid(*K, indexing='ij', sparse=True)
        Lp = 2 * np.pi / L
        for i in range(self.ndim):
            Ks[i] = (Ks[i] * Lp[i]).astype(float)
        K = [np.broadcast_to(k, self.fft.shape(True)) for k in Ks]
        K = np.array(K).astype(float)
        self.K2 = np.sum(K * K, 0, dtype=float)
        self.Ku = -self.K2 * self.Du
        self.Kv = -self.K2 * self.Dv

        # Need this for diagnostics
        self.dx = self.L / nvars[0]
        self.dy = self.L / nvars[1]

    def eval_f(self, u, t):
        """
        Routine to evaluate the right-hand side of the problem.

        Parameters
        ----------
        u : dtype_u
            Current values of the numerical solution.
        t : float
            Current time of the numerical solution is computed.

        Returns
        -------
        f : dtype_f
            The right-hand side of the problem.
        """

        f = self.dtype_f(self.init)

        if self.spectral:
            f.impl[..., 0] = self.Ku * u[..., 0]
            f.impl[..., 1] = self.Kv * u[..., 1]
            tmpu = newDistArray(self.fft, False)
            tmpv = newDistArray(self.fft, False)
            tmpu[:] = self.fft.backward(u[..., 0], tmpu)
            tmpv[:] = self.fft.backward(u[..., 1], tmpv)
            tmpfu = -tmpu * tmpv**2 + self.A * (1 - tmpu)
            tmpfv = tmpu * tmpv**2 - self.B * tmpv
            f.expl[..., 0] = self.fft.forward(tmpfu)
            f.expl[..., 1] = self.fft.forward(tmpfv)

        else:
            u_hat = self.fft.forward(u[..., 0])
            lap_u_hat = self.Ku * u_hat
            f.impl[..., 0] = self.fft.backward(lap_u_hat, f.impl[..., 0])
            u_hat = self.fft.forward(u[..., 1])
            lap_u_hat = self.Kv * u_hat
            f.impl[..., 1] = self.fft.backward(lap_u_hat, f.impl[..., 1])
            f.expl[..., 0] = -u[..., 0] * u[..., 1] ** 2 + self.A * (1 - u[..., 0])
            f.expl[..., 1] = u[..., 0] * u[..., 1] ** 2 - self.B * u[..., 1]

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

        me = self.dtype_u(self.init)
        if self.spectral:
            me[..., 0] = rhs[..., 0] / (1.0 - factor * self.Ku)
            me[..., 1] = rhs[..., 1] / (1.0 - factor * self.Kv)

        else:
            rhs_hat = self.fft.forward(rhs[..., 0])
            rhs_hat /= 1.0 - factor * self.Ku
            me[..., 0] = self.fft.backward(rhs_hat, me[..., 0])
            rhs_hat = self.fft.forward(rhs[..., 1])
            rhs_hat /= 1.0 - factor * self.Kv
            me[..., 1] = self.fft.backward(rhs_hat, me[..., 1])

        return me

    def u_exact(self, t):
        """
        Routine to compute the exact solution at time t=0, see https://www.chebfun.org/examples/pde/GrayScott.html

        Parameters
        ----------
        t : float
            Time of the exact solution.

        Returns
        -------
        me : dtype_u
            The exact solution.
        """
        assert t == 0.0, 'Exact solution only valid as initial condition'
        assert self.ndim == 2, 'The initial conditions are 2D for now..'

        me = self.dtype_u(self.init, val=0.0)

        # This assumes that the box is [-L/2, L/2]^2
        if self.spectral:
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
    def __init__(self, nvars, Du, Dv, A, B, spectral, L=2.0, comm=MPI.COMM_WORLD):
        super().__init__(nvars, Du, Dv, A, B, spectral, L, comm)
        self.Ku -= self.A
        self.Kv -= self.B

    def eval_f(self, u, t):
        """
        Routine to evaluate the right-hand side of the problem.

        Parameters
        ----------
        u : dtype_u
            Current values of the numerical solution.
        t : float
            Current time of the numerical solution is computed.

        Returns
        -------
        f : dtype_f
            The right-hand side of the problem.
        """

        f = self.dtype_f(self.init)

        if self.spectral:
            f.impl[..., 0] = self.Ku * u[..., 0]
            f.impl[..., 1] = self.Kv * u[..., 1]
            tmpu = newDistArray(self.fft, False)
            tmpv = newDistArray(self.fft, False)
            tmpu[:] = self.fft.backward(u[..., 0], tmpu)
            tmpv[:] = self.fft.backward(u[..., 1], tmpv)
            tmpfu = -tmpu * tmpv**2 + self.A
            tmpfv = tmpu * tmpv**2
            f.expl[..., 0] = self.fft.forward(tmpfu)
            f.expl[..., 1] = self.fft.forward(tmpfv)

        else:
            u_hat = self.fft.forward(u[..., 0])
            lap_u_hat = self.Ku * u_hat
            f.impl[..., 0] = self.fft.backward(lap_u_hat, f.impl[..., 0])
            u_hat = self.fft.forward(u[..., 1])
            lap_u_hat = self.Kv * u_hat
            f.impl[..., 1] = self.fft.backward(lap_u_hat, f.impl[..., 1])
            f.expl[..., 0] = -u[..., 0] * u[..., 1] ** 2 + self.A
            f.expl[..., 1] = u[..., 0] * u[..., 1] ** 2

        return f


class grayscott_mi_diffusion(grayscott_imex_diffusion):
    dtype_f = comp2_mesh

    def __init__(self, nvars, Du, Dv, A, B, spectral, newton_maxiter, newton_tol, L=2.0, comm=MPI.COMM_WORLD):
        super().__init__(nvars, Du, Dv, A, B, spectral, L, comm)
        # This may not run in parallel yet..
        assert self.comm.Get_size() == 1

    def eval_f(self, u, t):
        """
        Routine to evaluate the right-hand side of the problem.

        Parameters
        ----------
        u : dtype_u
            Current values of the numerical solution.
        t : float
            Current time of the numerical solution is computed.

        Returns
        -------
        f : dtype_f
            The right-hand side of the problem.
        """

        f = self.dtype_f(self.init)

        if self.spectral:
            f.comp1[..., 0] = self.Ku * u[..., 0]
            f.comp1[..., 1] = self.Kv * u[..., 1]
            tmpu = newDistArray(self.fft, False)
            tmpv = newDistArray(self.fft, False)
            tmpu[:] = self.fft.backward(u[..., 0], tmpu)
            tmpv[:] = self.fft.backward(u[..., 1], tmpv)
            tmpfu = -tmpu * tmpv**2 + self.A * (1 - tmpu)
            tmpfv = tmpu * tmpv**2 - self.B * tmpv
            f.comp2[..., 0] = self.fft.forward(tmpfu)
            f.comp2[..., 1] = self.fft.forward(tmpfv)

        else:
            u_hat = self.fft.forward(u[..., 0])
            lap_u_hat = self.Ku * u_hat
            f.comp1[..., 0] = self.fft.backward(lap_u_hat, f.comp1[..., 0])
            u_hat = self.fft.forward(u[..., 1])
            lap_u_hat = self.Kv * u_hat
            f.comp1[..., 1] = self.fft.backward(lap_u_hat, f.comp1[..., 1])
            f.comp2[..., 0] = -u[..., 0] * u[..., 1] ** 2 + self.A * (1 - u[..., 0])
            f.comp2[..., 1] = u[..., 0] * u[..., 1] ** 2 - self.B * u[..., 1]

        return f

    def solve_system_1(self, rhs, factor, u0, t):
        """
        Solver for the first component, can just call the super function.

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

        me = super(grayscott_mi_diffusion, self).solve_system(rhs, factor, u0, t)
        return me

    def solve_system_2(self, rhs, factor, u0, t):
        """
        Newton-Solver for the second component.

        Parameters
        ----------
        rhs : dtype_f
            Right-hand side for the linear system.
        factor float
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
        u = self.dtype_u(u0)

        if self.spectral:
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
        while n < self.newton_maxiter:
            # print(n, res)
            # form the function g with g(u) = 0
            tmpgu = tmpu - tmprhsu - factor * (-tmpu * tmpv**2 + self.A * (1 - tmpu))
            tmpgv = tmpv - tmprhsv - factor * (tmpu * tmpv**2 - self.B * tmpv)

            # if g is close to 0, then we are done
            res = max(np.linalg.norm(tmpgu, np.inf), np.linalg.norm(tmpgv, np.inf))
            if res < self.newton_tol:
                break

            # assemble dg
            dg00 = 1 - factor * (-(tmpv**2) - self.A)
            dg01 = -factor * (-2 * tmpu * tmpv)
            dg10 = -factor * (tmpv**2)
            dg11 = 1 - factor * (2 * tmpu * tmpv - self.B)

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
            tmpu[:] -= b[::2].reshape(self.nvars)
            tmpv[:] -= b[1::2].reshape(self.nvars)

            # increase iteration count
            n += 1

        if np.isnan(res) and self.stop_at_nan:
            raise ProblemError('Newton got nan after %i iterations, aborting...' % n)
        elif np.isnan(res):
            self.logger.warning('Newton got nan after %i iterations...' % n)

        if n == self.newton_maxiter:
            self.logger.warning('Newton did not converge after %i iterations, error is %s' % (n, res))

        # self.newton_ncalls += 1
        # self.newton_itercount += n
        me = self.dtype_u(self.init)
        if self.spectral:
            me[..., 0] = self.fft.forward(tmpu)
            me[..., 1] = self.fft.forward(tmpv)
        else:
            me[..., 0] = tmpu
            me[..., 1] = tmpv
        return me


class grayscott_mi_linear(grayscott_imex_linear):
    dtype_f = comp2_mesh

    def __init__(self, nvars, Du, Dv, A, B, spectral, newton_maxiter, newton_tol, L=2.0, comm=MPI.COMM_WORLD):
        super().__init__(nvars, Du, Dv, A, B, spectral, L, comm)
        # This may not run in parallel yet..
        assert self.comm.Get_size() == 1

    def eval_f(self, u, t):
        """
        Routine to evaluate the right-hand side of the problem.

        Parameters
        ----------
        u : dtype_u
            Current values of the numerical solution.
        t : float
            Current time of the numerical solution is computed.

        Returns
        -------
        f : dtype_f
            The right-hand side of the problem.
        """

        f = self.dtype_f(self.init)

        if self.spectral:
            f.comp1[..., 0] = self.Ku * u[..., 0]
            f.comp1[..., 1] = self.Kv * u[..., 1]
            tmpu = newDistArray(self.fft, False)
            tmpv = newDistArray(self.fft, False)
            tmpu[:] = self.fft.backward(u[..., 0], tmpu)
            tmpv[:] = self.fft.backward(u[..., 1], tmpv)
            tmpfu = -tmpu * tmpv**2 + self.A
            tmpfv = tmpu * tmpv**2
            f.comp2[..., 0] = self.fft.forward(tmpfu)
            f.comp2[..., 1] = self.fft.forward(tmpfv)

        else:
            u_hat = self.fft.forward(u[..., 0])
            lap_u_hat = self.Ku * u_hat
            f.comp1[..., 0] = self.fft.backward(lap_u_hat, f.comp1[..., 0])
            u_hat = self.fft.forward(u[..., 1])
            lap_u_hat = self.Kv * u_hat
            f.comp1[..., 1] = self.fft.backward(lap_u_hat, f.comp1[..., 1])
            f.comp2[..., 0] = -u[..., 0] * u[..., 1] ** 2 + self.A
            f.comp2[..., 1] = u[..., 0] * u[..., 1] ** 2

        return f

    def solve_system_1(self, rhs, factor, u0, t):
        """
        Solver for the first component, can just call the super function.

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

        me = super(grayscott_mi_linear, self).solve_system(rhs, factor, u0, t)
        return me

    def solve_system_2(self, rhs, factor, u0, t):
        """
        Newton-Solver for the second component.

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
        u : dtype_u
            The solution as mesh.
        """
        u = self.dtype_u(u0)

        if self.spectral:
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
        while n < self.newton_maxiter:
            # print(n, res)
            # form the function g with g(u) = 0
            tmpgu = tmpu - tmprhsu - factor * (-tmpu * tmpv**2 + self.A)
            tmpgv = tmpv - tmprhsv - factor * (tmpu * tmpv**2)

            # if g is close to 0, then we are done
            res = max(np.linalg.norm(tmpgu, np.inf), np.linalg.norm(tmpgv, np.inf))
            if res < self.newton_tol:
                break

            # assemble dg
            dg00 = 1 - factor * (-(tmpv**2))
            dg01 = -factor * (-2 * tmpu * tmpv)
            dg10 = -factor * (tmpv**2)
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
            tmpu[:] -= b[::2].reshape(self.nvars)
            tmpv[:] -= b[1::2].reshape(self.nvars)

            # increase iteration count
            n += 1

        if np.isnan(res) and self.stop_at_nan:
            raise ProblemError('Newton got nan after %i iterations, aborting...' % n)
        elif np.isnan(res):
            self.logger.warning('Newton got nan after %i iterations...' % n)

        if n == self.newton_maxiter:
            self.logger.warning('Newton did not converge after %i iterations, error is %s' % (n, res))

        # self.newton_ncalls += 1
        # self.newton_itercount += n
        me = self.dtype_u(self.init)
        if self.spectral:
            me[..., 0] = self.fft.forward(tmpu)
            me[..., 1] = self.fft.forward(tmpv)
        else:
            me[..., 0] = tmpu
            me[..., 1] = tmpv
        return me
