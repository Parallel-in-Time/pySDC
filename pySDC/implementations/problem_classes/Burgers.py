import numpy as np

from pySDC.implementations.datatype_classes.mesh import mesh, imex_mesh
from pySDC.implementations.problem_classes.generic_spectral import GenericSpectralLinear


class Burgers1D(GenericSpectralLinear):
    """
    1D viscous Burgers equation with Dirichlet BCs and a Chebychev method, IMEX with diffusion implicit, advection
    explicit.

    See https://en.wikipedia.org/wiki/Burgers'_equation for the equation that is solved.
    Discretization is done with a Chebychev method, which requires a first order derivative formulation.
    Feel free to do a more efficient implementation using an ultraspherical method to avoid the first order business.

    Parameters:
       N (int): Spatial resolution
       epsilon (float): viscosity
       BCl (float): Value at left boundary
       BCr (float): Value at right boundary
       f (int): Frequency of the initial conditions
       mode (str): 'T2U' or 'T2T'. Use 'T2U' to get sparse differentiation matrices
    """

    dtype_u = mesh
    dtype_f = imex_mesh

    def __init__(self, N=64, epsilon=0.1, BCl=1, BCr=-1, f=0, mode='T2U', **kwargs):
        """
        Constructor. `kwargs` are forwarded to parent class constructor.

        Args:
            N (int): Spatial resolution
            epsilon (float): viscosity
            BCl (float): Value at left boundary
            BCr (float): Value at right boundary
            f (int): Frequency of the initial conditions
            mode (str): 'T2U' or 'T2T'. Use 'T2U' to get sparse differentiation matrices
        """
        self._makeAttributeAndRegister('N', 'epsilon', 'BCl', 'BCr', 'f', 'mode', localVars=locals(), readOnly=True)

        bases = [{'base': 'cheby', 'N': N}]
        components = ['u', 'ux']

        super().__init__(bases=bases, components=components, spectral_space=False, **kwargs)

        self.x = self.get_grid()[0]

        # prepare matrices
        Dx = self.get_differentiation_matrix(axes=(0,))
        I = self.get_Id()

        T2U = self.get_basis_change_matrix(conv=mode)

        self.Dx = Dx

        # construct linear operator
        L_lhs = {'u': {'ux': -epsilon * (T2U @ Dx)}, 'ux': {'u': -T2U @ Dx, 'ux': T2U @ I}}
        self.setup_L(L_lhs)

        # construct mass matrix
        M_lhs = {'u': {'u': T2U @ I}}
        self.setup_M(M_lhs)

        # boundary conditions
        self.add_BC(component='u', equation='u', axis=0, x=1, v=BCr, kind='Dirichlet')
        self.add_BC(component='u', equation='ux', axis=0, x=-1, v=BCl, kind='Dirichlet')
        self.setup_BCs()

    def u_exact(self, t=0, *args, **kwargs):
        r"""
        Routine to return the initial conditions at :math:`t=0` or the stationary solution for :math:`t=\infty`.

        At :math:`t=0`, ``u`` is the linear function through the boundary values multiplied by :math:`\cos(\pi f x)`,
        and ``ux`` is its derivative. For ``t=np.inf``, which requires ``f=0`` and ``BCl=-BCr``, ``u`` is the stationary
        viscous shock connecting the boundary values and ``ux`` is left zero. Other times are not implemented.

        Parameters
        ----------
        t : float, optional
            Time, either ``0`` or ``np.inf``.
        *args, **kwargs
            Not used.

        Returns
        -------
        me : dtype_u
            The solution in physical space.
        """
        me = self.u_init

        # x = (self.x + 1) / 2
        # g = 4 * (1 + np.exp(-(4 * x + t)/self.epsilon/32))
        # g_x = 4 * np.exp(-(4 * x + t)/self.epsilon/32) * (-4/self.epsilon/32)

        # me[0] = 3./4. - 1./g
        # me[1] = 1/g**2 * g_x

        # return me

        if t == 0:
            me[self.index('u')][:] = ((self.BCr + self.BCl) / 2 + (self.BCr - self.BCl) / 2 * self.x) * np.cos(
                self.x * np.pi * self.f
            )
            me[self.index('ux')][:] = (self.BCr - self.BCl) / 2 * np.cos(self.x * np.pi * self.f) + (
                (self.BCr + self.BCl) / 2 + (self.BCr - self.BCl) / 2 * self.x
            ) * self.f * np.pi * -np.sin(self.x * np.pi * self.f)
        elif t == np.inf and self.f == 0 and self.BCl == -self.BCr:
            me[0] = (self.BCl * np.exp((self.BCr - self.BCl) / (2 * self.epsilon) * self.x) + self.BCr) / (
                np.exp((self.BCr - self.BCl) / (2 * self.epsilon) * self.x) + 1
            )
        else:
            raise NotImplementedError

        return me

    def eval_f(self, u, *args, **kwargs):
        r"""
        Routine to evaluate the right-hand side, split into diffusion and advection.

        The implicit part of the ``u`` equation is the diffusion :math:`\varepsilon \partial_x u_x`, computed by
        applying the Chebychev differentiation matrix to the auxiliary component ``ux``. The explicit part is the
        advection :math:`-u u_x`, computed pointwise. The algebraic ``ux`` equation gets zero in both parts.

        Parameters
        ----------
        u : dtype_u
            Current values of the numerical solution, in physical space.
        *args, **kwargs
            Not used, the right-hand side does not depend on time.

        Returns
        -------
        f : dtype_f
            The right-hand side, with ``impl`` and ``expl`` parts.
        """
        f = self.f_init
        iu, iux = self.index('u'), self.index('ux')

        u_hat = self.transform(u)

        Dx_u_hat = self.u_init_forward
        Dx_u_hat[iux] = (self.Dx @ u_hat[iux].flatten()).reshape(u_hat[iu].shape)

        f.impl[iu] = self.epsilon * self.itransform(Dx_u_hat)[iux].real
        f.expl[iu] = -u[iu] * u[iux]
        return f

    def get_fig(self):  # pragma: no cover
        """
        Get a figure suitable to plot the solution of this problem.

        Returns
        -------
        self.fig : matplotlib.pyplot.figure.Figure
        """
        import matplotlib.pyplot as plt

        self.fig, axs = plt.subplots(constrained_layout=True)
        return self.fig

    def plot(self, u, t=None, fig=None, comp='u'):  # pragma: no cover
        r"""
        Plot the solution.

        Parameters
        ----------
        u : dtype_u
            Solution to be plotted
        t : float
            Time to display at the top of the figure
        fig : matplotlib.pyplot.figure.Figure, optional
            Figure with the same structure as a figure generated by `self.get_fig`. If none is supplied, a new figure will be generated.
        comp : str, optional
            Name of the component to plot, ``'u'`` or ``'ux'``.

        Returns
        -------
        None
        """
        fig = self.get_fig() if fig is None else fig
        ax = fig.axes[0]

        ax.plot(self.x, u[self.index(comp)])

        if t is not None:
            fig.suptitle(f't = {t:.2e}')

        ax.set_xlabel(r'$x$')
        ax.set_ylabel(r'$u$')


class Burgers2D(GenericSpectralLinear):
    """
    2D viscous Burgers equation, periodic in x (FFT) and Dirichlet in z (Chebychev), diffusion implicit, advection
    explicit.

    See https://en.wikipedia.org/wiki/Burgers'_equation for the equation that is solved.
    This implementation is discretized with FFTs in x and Chebychev in z.

    Parameters:
       nx (int): Spatial resolution in x direction
       nz (int): Spatial resolution in z direction
       epsilon (float): viscosity
       fux (int): Frequency of the initial conditions in x-direction
       fuz (int): Frequency of the initial conditions in z-direction
       mode (str): 'T2U' or 'T2T'. Use 'T2U' to get sparse differentiation matrices

    The boundary conditions are fixed: u = 0 at z = -1 and z = 1, v = -1 at z = -1 and v = 1 at z = 1.
    """

    dtype_u = mesh
    dtype_f = imex_mesh

    def __init__(self, nx=64, nz=64, epsilon=0.1, fux=2, fuz=1, mode='T2U', **kwargs):
        """
         Constructor. `kwargs` are forwarded to parent class constructor.

         Args:
        nx (int): Spatial resolution in x direction
        nz (int): Spatial resolution in z direction
        epsilon (float): viscosity
        fux (int): Frequency of the initial conditions in x-direction
        fuz (int): Frequency of the initial conditions in z-direction
        mode (str): 'T2U' or 'T2T'. Use 'T2U' to get sparse differentiation matrices
        """
        self._makeAttributeAndRegister('nx', 'nz', 'epsilon', 'fux', 'fuz', 'mode', localVars=locals(), readOnly=True)

        bases = [
            {'base': 'fft', 'N': nx},
            {'base': 'cheby', 'N': nz},
        ]
        components = ['u', 'v', 'ux', 'uz', 'vx', 'vz']
        super().__init__(bases=bases, components=components, spectral_space=False, **kwargs)

        self.X, self.Z = self.get_grid()

        # prepare matrices
        Dx = self.get_differentiation_matrix(axes=(0,))
        Dz = self.get_differentiation_matrix(axes=(1,))
        I = self.get_Id()

        T2U = self.get_basis_change_matrix(axes=(1,), conv=mode)

        self.Dx = Dx
        self.Dz = Dz

        # construct linear operator
        L_lhs = {
            'u': {'ux': -epsilon * T2U @ Dx, 'uz': -epsilon * T2U @ Dz},
            'v': {'vx': -epsilon * T2U @ Dx, 'vz': -epsilon * T2U @ Dz},
            'ux': {'u': -T2U @ Dx, 'ux': T2U @ I},
            'uz': {'u': -T2U @ Dz, 'uz': T2U @ I},
            'vx': {'v': -T2U @ Dx, 'vx': T2U @ I},
            'vz': {'v': -T2U @ Dz, 'vz': T2U @ I},
        }
        self.setup_L(L_lhs)

        # construct mass matrix
        M_lhs = {
            'u': {'u': T2U @ I},
            'v': {'v': T2U @ I},
        }
        self.setup_M(M_lhs)

        # boundary conditions
        self.BCtop = 1
        self.BCbottom = -self.BCtop
        self.BCtopu = 0
        self.add_BC(component='v', equation='v', axis=1, v=self.BCtop, x=1, kind='Dirichlet')
        self.add_BC(component='v', equation='vz', axis=1, v=self.BCbottom, x=-1, kind='Dirichlet')
        self.add_BC(component='u', equation='uz', axis=1, v=self.BCtopu, x=1, kind='Dirichlet')
        self.add_BC(component='u', equation='u', axis=1, v=self.BCtopu, x=-1, kind='Dirichlet')
        self.setup_BCs()

    def u_exact(self, t=0, *args, noise_level=0, **kwargs):
        r"""
        Routine to return the initial conditions, which are only implemented for :math:`t=0`.

        ``u`` is :math:`\cos(f_{ux} x) \sin(\pi f_{uz} z)` plus ``BCtopu``, ``v`` is linear in :math:`z` between
        ``BCbottom`` and ``BCtop``, and the derivative components are set accordingly. Seeded normally distributed
        noise, multiplied by :math:`(z-1)(z+1)`, can be added to ``v``; its derivatives are not updated for the noise.

        Parameters
        ----------
        t : float, optional
            Time, has to be ``0``.
        *args, **kwargs
            Not used.
        noise_level : float, optional
            Amplitude of the noise added to ``v``.

        Returns
        -------
        me : dtype_u
            The initial conditions in physical space.
        """
        me = self.u_init

        iu, iv, iux, iuz, ivx, ivz = self.index(self.components)
        if t == 0:
            me[iu] = self.xp.cos(self.X * self.fux) * self.xp.sin(self.Z * np.pi * self.fuz) + self.BCtopu
            me[iux] = -self.xp.sin(self.X * self.fux) * self.fux * self.xp.sin(self.Z * np.pi * self.fuz)
            me[iuz] = self.xp.cos(self.X * self.fux) * self.xp.cos(self.Z * np.pi * self.fuz) * np.pi * self.fuz

            me[iv] = (self.BCtop + self.BCbottom) / 2 + (self.BCtop - self.BCbottom) / 2 * self.Z
            me[ivz][:] = (self.BCtop - self.BCbottom) / 2

            # add noise
            rng = self.xp.random.default_rng(seed=99)
            me[iv].real += rng.normal(size=me[iv].shape) * (self.Z - 1) * (self.Z + 1) * noise_level

        else:
            raise NotImplementedError

        return me

    def eval_f(self, u, *args, **kwargs):
        r"""
        Routine to evaluate the right-hand side, split into diffusion and advection.

        The implicit part is the diffusion :math:`\varepsilon (\partial_x w_x + \partial_z w_z)` of both velocity
        components :math:`w=u,v`, computed by spectral differentiation of the auxiliary derivative components. The
        explicit part is the advection :math:`-(u \partial_x w + v \partial_z w)`, computed pointwise. The algebraic
        derivative equations get zero in both parts.

        Parameters
        ----------
        u : dtype_u
            Current values of the numerical solution, in physical space.
        *args, **kwargs
            Not used, the right-hand side does not depend on time.

        Returns
        -------
        f : dtype_f
            The right-hand side, with ``impl`` and ``expl`` parts.
        """
        f = self.f_init
        iu, iv, iux, iuz, ivx, ivz = self.index(self.components)

        u_hat = self.transform(u)
        f_hat = self.u_init_forward
        f_hat[iu] = self.epsilon * ((self.Dx @ u_hat[iux].flatten() + self.Dz @ u_hat[iuz].flatten())).reshape(
            u_hat[iux].shape
        )
        f_hat[iv] = self.epsilon * ((self.Dx @ u_hat[ivx].flatten() + self.Dz @ u_hat[ivz].flatten())).reshape(
            u_hat[iux].shape
        )
        f.impl[...] = self.itransform(f_hat).real

        f.expl[iu] = -(u[iu] * u[iux] + u[iv] * u[iuz])
        f.expl[iv] = -(u[iu] * u[ivx] + u[iv] * u[ivz])
        return f

    def compute_vorticity(self, u):
        r"""
        Compute the vorticity from the velocity components by spectral differentiation, as :math:`\partial_x v +
        \partial_z u`.

        Parameters
        ----------
        u : dtype_u
            Solution in physical space.

        Returns
        -------
        vorticity : np.2darray
            The vorticity in physical space.
        """
        me = self.u_init_forward

        u_hat = self.transform(u)
        iu, iv = self.index(['u', 'v'])

        me[iu] = (self.Dx @ u_hat[iv].flatten() - self.Dz @ u_hat[iu].flatten()).reshape(u[iu].shape)
        return self.itransform(me)[iu].real

    def get_fig(self):  # pragma: no cover
        """
        Get a figure suitable to plot the solution of this problem

        Returns
        -------
        self.fig : matplotlib.pyplot.figure.Figure
        """
        import matplotlib.pyplot as plt
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        self.fig, axs = plt.subplots(3, 1, sharex=True, sharey=True, figsize=((8, 7)), constrained_layout=True)
        self.cax = []
        divider = make_axes_locatable(axs[0])
        self.cax += [divider.append_axes('right', size='3%', pad=0.03)]
        divider2 = make_axes_locatable(axs[1])
        self.cax += [divider2.append_axes('right', size='3%', pad=0.03)]
        divider3 = make_axes_locatable(axs[2])
        self.cax += [divider3.append_axes('right', size='3%', pad=0.03)]
        return self.fig

    def plot(self, u, t=None, fig=None, vmin=None, vmax=None):  # pragma: no cover
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
        vmin : float, optional
            Lower limit of the colour scale for the velocity components.
        vmax : float, optional
            Upper limit of the colour scale for the velocity components.

        Returns
        -------
        None
        """
        fig = self.get_fig() if fig is None else fig
        axs = fig.axes

        iu, iv = self.index(['u', 'v'])

        imU = axs[0].pcolormesh(self.X, self.Z, u[iu].real, vmin=vmin, vmax=vmax)
        imV = axs[1].pcolormesh(self.X, self.Z, u[iv].real, vmin=vmin, vmax=vmax)
        imVort = axs[2].pcolormesh(self.X, self.Z, self.compute_vorticity(u).real)

        for i, label in zip([0, 1, 2], [r'$u$', '$v$', 'vorticity'], strict=True):
            axs[i].set_aspect(1)
            axs[i].set_title(label)

        if t is not None:
            fig.suptitle(f't = {t:.2e}')
        axs[-1].set_xlabel(r'$x$')
        axs[-1].set_ylabel(r'$z$')
        fig.colorbar(imU, self.cax[0])
        fig.colorbar(imV, self.cax[1])
        fig.colorbar(imVort, self.cax[2])
