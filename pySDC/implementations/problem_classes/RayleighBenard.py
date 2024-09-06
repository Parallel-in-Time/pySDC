import numpy as np
from mpi4py import MPI

from pySDC.implementations.problem_classes.generic_spectral import GenericSpectralLinear
from pySDC.implementations.datatype_classes.mesh import mesh, imex_mesh
from pySDC.core.convergence_controller import ConvergenceController
from pySDC.core.hooks import Hooks
from pySDC.implementations.convergence_controller_classes.check_convergence import CheckConvergence


class RayleighBenard(GenericSpectralLinear):
    """
    Rayleigh-Benard Convection is a variation of incompressible Navier-Stokes. See, for instance https://doi.org/10.1007/s00791-020-00332-3.

    Parameters:
        Prandl (float): Prandl number
        Rayleigh (float): Rayleigh number
        nx (int): Horizontal resolution
        nz (int): Vertical resolution
        BCs (dict): Can specify boundary conditions here
        dealiasing (float): Dealiasing factor for evaluating the non-linear part
        comm (mpi4py.Intracomm): Space communicator
    """

    dtype_u = mesh
    dtype_f = imex_mesh

    def __init__(
        self,
        Prandl=1,
        Rayleigh=2e6,
        nx=257,
        nz=64,
        BCs=None,
        dealiasing=3 / 2,
        comm=None,
        **kwargs,
    ):
        BCs = {} if BCs is None else BCs
        BCs = {
            'T_top': 0,
            'T_bottom': 2,
            'v_top': 0,
            'v_bottom': 0,
            'u_top': 0,
            'u_bottom': 0,
            'p_integral': 0,
            **BCs,
        }
        if comm is None:
            try:
                from mpi4py import MPI

                comm = MPI.COMM_WORLD
            except ModuleNotFoundError:
                pass
        self._makeAttributeAndRegister(
            'Prandl',
            'Rayleigh',
            'nx',
            'nz',
            'BCs',
            'dealiasing',
            'comm',
            localVars=locals(),
            readOnly=True,
        )

        bases = [{'base': 'fft', 'N': nx, 'x0': 0, 'x1': 8}, {'base': 'ultraspherical', 'N': nz}]
        components = ['u', 'v', 'T', 'p']
        super().__init__(bases, components, comm=comm, **kwargs)

        self.Z, self.X = self.get_grid()
        self.Kz, self.Kx = self.get_wavenumbers()

        # construct 2D matrices
        Dzz = self.get_differentiation_matrix(axes=(1,), p=2)
        Dz = self.get_differentiation_matrix(axes=(1,))
        Dx = self.get_differentiation_matrix(axes=(0,))
        Dxx = self.get_differentiation_matrix(axes=(0,), p=2)
        Id = self.get_Id()

        S1 = self.get_basis_change_matrix(p_out=0, p_in=1)
        S2 = self.get_basis_change_matrix(p_out=0, p_in=2)

        U01 = self.get_basis_change_matrix(p_in=0, p_out=1)
        U12 = self.get_basis_change_matrix(p_in=1, p_out=2)
        U02 = self.get_basis_change_matrix(p_in=0, p_out=2)

        self.Dx = Dx
        self.Dxx = Dxx
        self.Dz = S1 @ Dz
        self.Dzz = S2 @ Dzz

        kappa = (Rayleigh * Prandl) ** (-1 / 2.0)
        nu = (Rayleigh / Prandl) ** (-1 / 2.0)

        # construct operators
        L_lhs = {
            'p': {'u': U01 @ Dx, 'v': Dz},  # divergence free constraint
            'u': {'p': U02 @ Dx, 'u': -nu * (U02 @ Dxx + Dzz)},
            'v': {'p': U12 @ Dz, 'v': -nu * (U02 @ Dxx + Dzz), 'T': -U02 @ Id},
            'T': {'T': -kappa * (U02 @ Dxx + Dzz)},
        }
        self.setup_L(L_lhs)

        # mass matrix
        M_lhs = {i: {i: U02 @ Id} for i in ['u', 'v', 'T']}
        self.setup_M(M_lhs)

        self.base_change = self._setup_operator({**{comp: {comp: S2} for comp in ['u', 'v', 'T']}, 'p': {'p': S1}})

        self.add_BC(
            component='p', equation='p', axis=1, v=self.BCs['p_integral'], kind='integral', line=-1, scalar=True
        )
        self.add_BC(component='T', equation='T', axis=1, x=-1, v=self.BCs['T_bottom'], kind='Dirichlet', line=-1)
        self.add_BC(component='T', equation='T', axis=1, x=1, v=self.BCs['T_top'], kind='Dirichlet', line=-2)
        self.add_BC(component='v', equation='v', axis=1, x=1, v=self.BCs['v_bottom'], kind='Dirichlet', line=-1)
        self.add_BC(component='v', equation='v', axis=1, x=-1, v=self.BCs['v_bottom'], kind='Dirichlet', line=-2)
        self.remove_BC(component='v', equation='v', axis=1, x=-1, kind='Dirichlet', line=-2, scalar=True)
        self.add_BC(component='u', equation='u', axis=1, v=self.BCs['u_top'], x=1, kind='Dirichlet', line=-2)
        self.add_BC(
            component='u',
            equation='u',
            axis=1,
            v=self.BCs['u_bottom'],
            x=-1,
            kind='Dirichlet',
            line=-1,
        )
        self.setup_BCs()

        if nx % 2 == 0:
            self.logger.warning(
                f'The resolution is x-direction is even at {nx}. Keep in mind that the Nyquist mode is only partially resolved in this case. Consider changing the solution by one.'
            )

    def eval_f(self, u, *args, **kwargs):
        f = self.f_init

        u_hat = self.transform(u)
        f_impl_hat = self.u_init_forward

        Dz = self.Dz
        Dx = self.Dx

        iu, iv, iT, ip = self.index(['u', 'v', 'T', 'p'])

        # evaluate implicit terms
        f_impl_hat = -(self.base_change @ self.L @ u_hat.flatten()).reshape(u_hat.shape)
        f.impl[:] = self.itransform(f_impl_hat).real

        # treat convection explicitly with dealiasing
        Dx_u_hat = self.u_init_forward
        for i in [iu, iv, iT]:
            Dx_u_hat[i][:] = (Dx @ u_hat[i].flatten()).reshape(Dx_u_hat[i].shape)
        Dz_u_hat = self.u_init_forward
        for i in [iu, iv, iT]:
            Dz_u_hat[i][:] = (Dz @ u_hat[i].flatten()).reshape(Dz_u_hat[i].shape)

        padding = [self.dealiasing, self.dealiasing]
        Dx_u_pad = self.itransform(Dx_u_hat, padding=padding).real
        Dz_u_pad = self.itransform(Dz_u_hat, padding=padding).real
        u_pad = self.itransform(u_hat, padding=padding).real

        fexpl_pad = self.xp.zeros_like(u_pad)
        fexpl_pad[iu][:] = -(u_pad[iu] * Dx_u_pad[iu] + u_pad[iv] * Dz_u_pad[iu])
        fexpl_pad[iv][:] = -(u_pad[iu] * Dx_u_pad[iv] + u_pad[iv] * Dz_u_pad[iv])
        fexpl_pad[iT][:] = -(u_pad[iu] * Dx_u_pad[iT] + u_pad[iv] * Dz_u_pad[iT])

        f.expl[:] = self.itransform(self.transform(fexpl_pad, padding=padding)).real

        return f

    def u_exact(self, t=0, noise_level=1e-3, seed=99):
        assert t == 0
        assert (
            self.BCs['v_top'] == self.BCs['v_bottom']
        ), 'Initial conditions are only implemented for zero velocity gradient'

        me = self.u_init
        iu, iv, iT, ip = self.index(['u', 'v', 'T', 'p'])

        # linear temperature gradient
        for comp in ['T', 'v', 'u']:
            a = (self.BCs[f'{comp}_top'] - self.BCs[f'{comp}_bottom']) / 2
            b = (self.BCs[f'{comp}_top'] + self.BCs[f'{comp}_bottom']) / 2
            me[self.index(comp)] = a * self.Z + b

        # perturb slightly
        rng = self.xp.random.default_rng(seed=seed)

        noise = self.u_init
        noise[iT] = rng.random(size=me[iT].shape)

        me[iT] += noise[iT].real * noise_level * (self.Z - 1) * (self.Z + 1)

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
        self.fig, axs = plt.subplots(2, 1, sharex=True, sharey=True, figsize=((10, 5)))
        self.cax = []
        divider = make_axes_locatable(axs[0])
        self.cax += [divider.append_axes('right', size='3%', pad=0.03)]
        divider2 = make_axes_locatable(axs[1])
        self.cax += [divider2.append_axes('right', size='3%', pad=0.03)]
        return self.fig

    def plot(self, u, t=None, fig=None, quantity='T'):  # pragma: no cover
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

        imT = axs[0].pcolormesh(self.X, self.Z, u[self.index(quantity)].real)
        imV = axs[1].pcolormesh(self.X, self.Z, self.compute_vorticity(u).real)

        for i, label in zip([0, 1], [rf'${quantity}$', 'vorticity']):
            axs[i].set_aspect(1)
            axs[i].set_title(label)

        if t is not None:
            fig.suptitle(f't = {t:.2f}')
        axs[1].set_xlabel(r'$x$')
        axs[1].set_ylabel(r'$z$')
        fig.colorbar(imT, self.cax[0])
        fig.colorbar(imV, self.cax[1])

    def compute_vorticity(self, u):
        u_hat = self.transform(u)
        Dz = self.Dz
        Dx = self.Dx
        iu, iv = self.index(['u', 'v'])

        vorticity_hat = self.u_init_forward
        vorticity_hat[0] = (Dx * u_hat[iv].flatten() + Dz @ u_hat[iu].flatten()).reshape(u[iu].shape)
        return self.itransform(vorticity_hat)[0].real

    def compute_Nusselt_numbers(self, u):
        """
        Compute the various versions of the Nusselt number. This reflects the type of heat transport.
        If the Nusselt number is equal to one, it indicates heat transport due to conduction. If it is larger,
        advection is present.
        Computing the Nusselt number at various places can be used to check the code.

        Args:
            u: The solution you want to compute the Nusselt numbers of

        Returns:
            dict: Nusselt number averaged over the entire volume and horizontally averaged at the top and bottom.
        """
        iv, iT = self.index(['v', 'T'])

        DzT_hat = self.u_init_forward

        u_hat = self.transform(u)
        DzT_hat[iT] = (self.Dz @ u_hat[iT].flatten()).reshape(DzT_hat[iT].shape)

        # compute vT with dealiasing
        padding = [self.dealiasing, self.dealiasing]
        u_pad = self.itransform(u_hat, padding=padding).real
        _me = self.xp.zeros_like(u_pad)
        _me[0] = u_pad[iv] * u_pad[iT]
        vT_hat = self.transform(_me, padding=padding)

        nusselt_hat = (vT_hat[0] - DzT_hat[iT]) / self.nx
        nusselt_no_v_hat = (-DzT_hat[iT]) / self.nx

        integral_z = self.xp.sum(nusselt_hat * self.spectral.axes[1].get_BC(kind='integral'), axis=-1).real
        integral_V = (
            integral_z[0] * self.axes[0].L
        )  # only the first Fourier mode has non-zero integral with periodic BCs
        Nusselt_V = self.comm.bcast(integral_V / self.spectral.V, root=0)

        Nusselt_t = self.comm.bcast(
            self.xp.sum(nusselt_hat * self.spectral.axes[1].get_BC(kind='Dirichlet', x=1), axis=-1).real[0], root=0
        )
        Nusselt_b = self.comm.bcast(
            self.xp.sum(nusselt_hat * self.spectral.axes[1].get_BC(kind='Dirichlet', x=-1), axis=-1).real[0], root=0
        )
        Nusselt_no_v_t = self.comm.bcast(
            self.xp.sum(nusselt_no_v_hat * self.spectral.axes[1].get_BC(kind='Dirichlet', x=1), axis=-1).real[0], root=0
        )
        Nusselt_no_v_b = self.comm.bcast(
            self.xp.sum(nusselt_no_v_hat * self.spectral.axes[1].get_BC(kind='Dirichlet', x=-1), axis=-1).real[0],
            root=0,
        )

        return {
            'V': Nusselt_V,
            't': Nusselt_t,
            'b': Nusselt_b,
            't_no_v': Nusselt_no_v_t,
            'b_no_v': Nusselt_no_v_b,
        }

    def compute_viscous_dissipation(self, u):
        iu, iv = self.index(['u', 'v'])

        Lap_u_hat = self.u_init_forward

        u_hat = self.transform(u)
        Lap_u_hat[iu] = ((self.Dzz + self.Dxx) @ u_hat[iu].flatten()).reshape(u_hat[iu].shape)
        Lap_u_hat[iv] = ((self.Dzz + self.Dxx) @ u_hat[iv].flatten()).reshape(u_hat[iu].shape)
        Lap_u = self.itransform(Lap_u_hat)

        return abs(u[iu] * Lap_u[iu] + u[iv] * Lap_u[iv])

    def compute_buoyancy_generation(self, u):
        iv, iT = self.index(['v', 'T'])
        return abs(u[iv] * self.Rayleigh * u[iT])


class CFLLimit(ConvergenceController):

    def dependencies(self, controller, *args, **kwargs):
        from pySDC.implementations.hooks.log_step_size import LogStepSize

        controller.add_hook(LogCFL)
        controller.add_hook(LogStepSize)

    def setup_status_variables(self, controller, **kwargs):
        """
        Add the embedded error variable to the error function.

        Args:
            controller (pySDC.Controller): The controller
        """
        self.add_status_variable_to_level('CFL_limit')

    def setup(self, controller, params, description, **kwargs):
        """
        Define default parameters here.

        Default parameters are:
         - control_order (int): The order relative to other convergence controllers
         - dt_max (float): maximal step size
         - dt_min (float): minimal step size

        Args:
            controller (pySDC.Controller): The controller
            params (dict): The params passed for this specific convergence controller
            description (dict): The description object used to instantiate the controller

        Returns:
            (dict): The updated params dictionary
        """
        defaults = {
            "control_order": -50,
            "dt_max": np.inf,
            "dt_min": 0,
            "cfl": 0.4,
        }
        return {**defaults, **super().setup(controller, params, description, **kwargs)}

    @staticmethod
    def compute_max_step_size(P, u):
        grid_spacing_x = P.X[1, 0] - P.X[0, 0]

        cell_wallz = P.xp.zeros(P.nz + 1)
        cell_wallz[0] = 1
        cell_wallz[-1] = -1
        cell_wallz[1:-1] = (P.Z[0, :-1] + P.Z[0, 1:]) / 2
        grid_spacing_z = cell_wallz[:-1] - cell_wallz[1:]

        iu, iv = P.index(['u', 'v'])

        max_step_size_x = P.xp.min(grid_spacing_x / P.xp.abs(u[iu]))
        max_step_size_z = P.xp.min(grid_spacing_z / P.xp.abs(u[iv]))
        max_step_size = min([max_step_size_x, max_step_size_z])

        if hasattr(P, 'comm'):
            max_step_size = P.comm.allreduce(max_step_size, op=MPI.MIN)
        return max_step_size

    def get_new_step_size(self, controller, step, **kwargs):
        if not CheckConvergence.check_convergence(step):
            return None

        L = step.levels[0]
        P = step.levels[0].prob

        L.sweep.compute_end_point()
        max_step_size = self.compute_max_step_size(P, L.uend)

        L.status.CFL_limit = self.params.cfl * max_step_size

        dt_new = L.status.dt_new if L.status.dt_new else max([self.params.dt_max, L.params.dt])
        L.status.dt_new = min([dt_new, self.params.cfl * max_step_size])
        L.status.dt_new = max([self.params.dt_min, L.status.dt_new])

        self.log(f'dt max: {max_step_size:.2e} -> New step size: {L.status.dt_new:.2e}', step)


class LogCFL(Hooks):

    def post_step(self, step, level_number):
        """
        Record CFL limit.

        Args:
            step (pySDC.Step.step): the current step
            level_number (int): the current level number

        Returns:
            None
        """
        super().post_step(step, level_number)

        L = step.levels[level_number]

        self.add_to_stats(
            process=step.status.slot,
            time=L.time + L.dt,
            level=L.level_index,
            iter=step.status.iter,
            sweep=L.status.sweep,
            type='CFL_limit',
            value=L.status.CFL_limit,
        )


class LogAnalysisVariables(Hooks):

    def post_step(self, step, level_number):
        """
        Record Nusselt numbers.

        Args:
            step (pySDC.Step.step): the current step
            level_number (int): the current level number

        Returns:
            None
        """
        super().post_step(step, level_number)

        L = step.levels[level_number]
        P = L.prob

        L.sweep.compute_end_point()
        Nusselt = P.compute_Nusselt_numbers(L.uend)
        buoyancy_production = P.compute_buoyancy_generation(L.uend)
        viscous_dissipation = P.compute_viscous_dissipation(L.uend)

        for key, value in zip(
            ['Nusselt', 'buoyancy_production', 'viscous_dissipation'],
            [Nusselt, buoyancy_production, viscous_dissipation],
        ):
            self.add_to_stats(
                process=step.status.slot,
                time=L.time + L.dt,
                level=L.level_index,
                iter=step.status.iter,
                sweep=L.status.sweep,
                type=key,
                value=value,
            )
