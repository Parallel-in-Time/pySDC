import numpy as np
from mpi4py import MPI

from pySDC.implementations.problem_classes.generic_MPIFFT_Laplacian import IMEX_Laplacian_MPIFFT


class Brusselator(IMEX_Laplacian_MPIFFT):
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

    def __init__(self, alpha=0.1, **kwargs):
        """Initialization routine"""
        super().__init__(spectral=False, L=1.0, dtype='d', alpha=alpha, **kwargs)

        # prepare the array with two components
        shape = (2,) + (self.init[0])
        self.iU = 0
        self.iV = 1
        self.ncomp = 2  # needed for transfer class
        self.init = (shape, self.comm, np.dtype('float'))

    def _eval_explicit_part(self, u, t, f_expl):
        iU, iV = self.iU, self.iV
        x, y = self.X[0], self.X[1]

        # evaluate time independent part
        f_expl[iU, ...] = 1.0 + u[iU] ** 2 * u[iV] - 4.4 * u[iU]
        f_expl[iV, ...] = 3.4 * u[iU] - u[iU] ** 2 * u[iV]

        # add time-dependent part
        if t >= 1.1:
            mask = (x - 0.3) ** 2 + (y - 0.6) ** 2 <= 0.1**2
            f_expl[iU][mask] += 5.0
        return f_expl

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

        # evaluate Laplacian to be solved implicitly
        for i in [self.iU, self.iV]:
            f.impl[i, ...] = self._eval_Laplacian(u[i], f.impl[i])

        f.expl[:] = self._eval_explicit_part(u, t, f.expl)

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
            me[i, ...] = self._invert_Laplacian(me[i], factor, rhs[i])

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
            me[iU, ...] = 22.0 * y * (1 - y / self.L[0]) ** (3.0 / 2.0) / self.L[0]
            me[iV, ...] = 27.0 * x * (1 - x / self.L[0]) ** (3.0 / 2.0) / self.L[0]
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
