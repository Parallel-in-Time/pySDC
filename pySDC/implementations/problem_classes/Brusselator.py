import numpy as np
from mpi4py import MPI
from mpi4py_fft import PFFT

from pySDC.core.Errors import ProblemError
from pySDC.core.Problem import ptype, WorkCounter
from pySDC.implementations.datatype_classes.mesh import mesh, imex_mesh

from mpi4py_fft import newDistArray


class Brusselator(ptype):
    r"""
    Two-dimensional Brusselator from [1]_.
    This is a reaction-diffusion equation with non-autonomous source term:

    .. math::
        \frac{\partial u}{\partial t} = \varalpha \Delta u + 1 + u^2 v - 4.4u _ f(x,y,t),
        \frac{\partial v}{\partial t} = \varalpha \Delta v + 3.4u - u^2 v

    with the source term :math:`f(x,y,t) = 5` if :math:`(x-0.3)^2 + (y-0.6)^2 <= 0.1^2` and :math:`t >= 1.1` and 0 else.
    We discretize in a periodic domain of length 1 and solve with an IMEX scheme based on a spectral method for the
    Laplacian which we invert implicitly. We treat the reaction and source terms explicitly.

    References
    ----------
    .. [1] https://link.springer.com/book/10.1007/978-3-642-05221-7
    """

    dtype_u = mesh
    dtype_f = imex_mesh

    def __init__(self, nvars=None, alpha=0.1, comm=MPI.COMM_WORLD):
        """Initialization routine"""
        nvars = (128,) * 2 if nvars is None else nvars
        L = 1.0

        if not (isinstance(nvars, tuple) and len(nvars) > 1):
            raise ProblemError('Need at least two dimensions')

        # Create FFT structure
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
        tmp_u = newDistArray(self.fft, False)

        # prepare the array with two components
        shape = (2,) + tmp_u.shape
        self.iU = 0
        self.iV = 1

        super().__init__(init=(shape, comm, tmp_u.dtype))
        self._makeAttributeAndRegister('nvars', 'alpha', 'L', 'comm', localVars=locals(), readOnly=True)

        L = np.array([self.L] * self.ndim, dtype=float)

        # get local mesh for distributed FFT
        X = np.ogrid[self.fft.local_slice(False)]
        N = self.fft.global_shape()
        for i in range(len(N)):
            X[i] = X[i] * L[i] / N[i]
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

        # Need this for diagnostics
        self.dx = self.L / nvars[0]
        self.dy = self.L / nvars[1]

        self.work_counters['rhs'] = WorkCounter()

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
        iU, iV = self.iU, self.iV
        x, y = self.X[0], self.X[1]

        f = self.dtype_f(self.init)

        # evaluate Laplacian to be solved implicitly
        for i in [self.iU, self.iV]:
            u_hat = self.fft.forward(u[i, ...])
            lap_u_hat = -self.alpha * self.K2 * u_hat
            f.impl[i, ...] = self.fft.backward(lap_u_hat, f.impl[i, ...])

        # evaluate time independent part
        f.expl[iU, ...] = 1.0 + u[iU] ** 2 * u[iV] - 4.4 * u[iU]
        f.expl[iV, ...] = 3.4 * u[iU] - u[iU] ** 2 * u[iV]

        # add time-dependent part
        if t >= 1.1:
            mask = (x - 0.3) ** 2 + (y - 0.6) ** 2 <= 0.1**2
            f.expl[iU][mask] += 5.0

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
            Solution.
        """
        me = self.dtype_u(self.init)

        for i in [self.iU, self.iV]:
            rhs_hat = self.fft.forward(rhs[i, ...])
            rhs_hat /= 1.0 + factor * self.K2 * self.alpha
            me[i, ...] = self.fft.backward(rhs_hat, me[i, ...])

        return me

    def u_exact(self, t, u_init=None, t_init=None):
        r"""
        Initial conditions.

        Parameters
        ----------
        t : float
            Time of the exact solution.
        u_init : dtype_u
            Initial conditions for getting the exact solution.
        t_init : float
            The starting time.

        Returns
        -------
        me : dtype_u
            Exact solution.
        """

        iU, iV = self.iU, self.iV
        x, y = self.X[0], self.X[1]

        me = self.dtype_u(self.init, val=0.0)

        if t == 0:
            me[iU, ...] = 22.0 * y * (1 - y / self.L) ** (3.0 / 2.0) / self.L
            me[iV, ...] = 27.0 * x * (1 - x / self.L) ** (3.0 / 2.0) / self.L
        else:

            def eval_rhs(t, u):
                f = self.eval_f(u.reshape(self.init[0]), t)
                return (f.impl + f.expl).flatten()

            me[...] = self.generate_scipy_reference_solution(eval_rhs, t, u_init, t_init)

        return me

    def get_fig(self):  # pragma: no cover
        """
        Get a figure suitable to plot the solution of this problem

        Returns
        -------
        self.fig : matplotlib.pyplot.figure.Figure
        """
        import matplotlib.pyplot as plt
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        plt.rcParams['figure.constrained_layout.use'] = True
        self.fig, axs = plt.subplots(1, 2, sharex=True, sharey=True, figsize=((8, 3)))
        divider = make_axes_locatable(axs[1])
        self.cax = divider.append_axes('right', size='3%', pad=0.03)
        return self.fig

    def plot(self, u, t=None, fig=None):  # pragma: no cover
        r"""
        Plot the solution. Please supply a figure with the same structure as returned by ``self.get_fig``.

        Parameters
        ----------
        u : dtype_u
            Solution to be plotted
        t : float
            Time to display at the top of the figure
        fig : matplotlib.pyplot.figure.Figure
            Figure with the correct structure

        Returns
        -------
        None
        """
        fig = self.get_fig() if fig is None else fig
        axs = fig.axes

        vmin = u.min()
        vmax = u.max()
        for i, label in zip([self.iU, self.iV], [r'$u$', r'$v$']):
            im = axs[i].pcolormesh(self.X[0], self.X[1], u[i], vmin=vmin, vmax=vmax)
            axs[i].set_aspect(1)
            axs[i].set_title(label)

        if t is not None:
            fig.suptitle(f't = {t:.2e}')
        axs[0].set_xlabel(r'$x$')
        axs[0].set_ylabel(r'$y$')
        fig.colorbar(im, self.cax)
