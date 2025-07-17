import numpy as np
from mpi4py import MPI

from pySDC.implementations.problem_classes.generic_spectral import GenericSpectralLinear
from pySDC.implementations.datatype_classes.mesh import mesh, imex_mesh
from pySDC.core.convergence_controller import ConvergenceController
from pySDC.core.hooks import Hooks
from pySDC.implementations.convergence_controller_classes.check_convergence import CheckConvergence
from pySDC.core.problem import WorkCounter


class RayleighBenard3D(GenericSpectralLinear):
    """
    Rayleigh-Benard Convection is a variation of incompressible Navier-Stokes.

    The equations we solve are

        u_x + v_y + w_z = 0
        T_t - kappa (T_xx + T_yy + T_zz) = -uT_x - vT_y - wT_z
        u_t - nu (u_xx + u_yy + u_zz) + p_x = -uu_x - vu_y - wu_z
        v_t - nu (v_xx + v_yy + v_zz) + p_y = -uv_x - vv_y - wv_z
        w_t - nu (w_xx + w_yy + w_zz) + p_z - T = -uw_x - vw_y - ww_z

    with u the horizontal velocity, v the vertical velocity (in z-direction), T the temperature, p the pressure, indices
    denoting derivatives, kappa=(Rayleigh * Prandtl)**(-1/2) and nu = (Rayleigh / Prandtl)**(-1/2). Everything on the left
    hand side, that is the viscous part, the pressure gradient and the buoyancy due to temperature are treated
    implicitly, while the non-linear convection part on the right hand side is integrated explicitly.

    The domain, vertical boundary conditions and pressure gauge are

        Omega = [0, 8) x (-1, 1)
        T(z=+1) = 0
        T(z=-1) = 2
        u(z=+-1) = v(z=+-1) = 0
        integral over p = 0

    The spectral discretization uses FFT horizontally, implying periodic BCs, and an ultraspherical method vertically to
    facilitate the Dirichlet BCs.

    Parameters:
        Prandtl (float): Prandtl number
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
        Prandtl=1,
        Rayleigh=2e6,
        nx=256,
        ny=256,
        nz=64,
        BCs=None,
        dealiasing=1.5,
        comm=None,
        Lz=1,
        Lx=1,
        Ly=1,
        useGPU=False,
        **kwargs,
    ):
        """
        Constructor. `kwargs` are forwarded to parent class constructor.

        Args:
            Prandtl (float): Prandtl number
            Rayleigh (float): Rayleigh number
            nx (int): Resolution in x-direction
            nz (int): Resolution in z direction
            BCs (dict): Vertical boundary conditions
            dealiasing (float): Dealiasing for evaluating the non-linear part in real space
            comm (mpi4py.Intracomm): Space communicator
            Lx (float): Horizontal length of the domain
        """
        # TODO: documentation
        BCs = {} if BCs is None else BCs
        BCs = {
            'T_top': 0,
            'T_bottom': Lz,
            'w_top': 0,
            'w_bottom': 0,
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
            'Prandtl',
            'Rayleigh',
            'nx',
            'ny',
            'nz',
            'BCs',
            'dealiasing',
            'comm',
            'Lx',
            'Ly',
            'Lz',
            localVars=locals(),
            readOnly=True,
        )

        bases = [
            {'base': 'fft', 'N': nx, 'x0': 0, 'x1': self.Lx, 'useFFTW': not useGPU},
            {'base': 'fft', 'N': ny, 'x0': 0, 'x1': self.Ly, 'useFFTW': not useGPU},
            {'base': 'ultraspherical', 'N': nz, 'x0': 0, 'x1': self.Lz},
        ]
        components = ['u', 'v', 'w', 'T', 'p']
        super().__init__(bases, components, comm=comm, useGPU=useGPU, **kwargs)

        self.X, self.Y, self.Z = self.get_grid()
        self.Kx, self.Ky, self.Kz = self.get_wavenumbers()

        # construct 3D matrices
        Dzz = self.get_differentiation_matrix(axes=(2,), p=2)
        Dz = self.get_differentiation_matrix(axes=(2,))
        Dy = self.get_differentiation_matrix(axes=(1,))
        Dyy = self.get_differentiation_matrix(axes=(1,), p=2)
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
        self.Dy = Dy
        self.Dyy = Dyy
        self.Dz = S1 @ Dz
        self.Dzz = S2 @ Dzz
        self.S2 = S2
        self.S1 = S1

        # compute rescaled Rayleigh number to extract viscosity and thermal diffusivity
        Ra = Rayleigh / (max([abs(BCs['T_top'] - BCs['T_bottom']), np.finfo(float).eps]) * self.axes[2].L ** 3)
        self.kappa = (Ra * Prandtl) ** (-1 / 2.0)
        self.nu = (Ra / Prandtl) ** (-1 / 2.0)

        # construct operators
        _D = U02 @ (Dxx + Dyy) + Dzz
        L_lhs = {
            'p': {'u': U01 @ Dx, 'v': U01 @ Dy, 'w': Dz},  # divergence free constraint
            'u': {'p': U02 @ Dx, 'u': -self.nu * _D},
            'v': {'p': U02 @ Dy, 'v': -self.nu * _D},
            'w': {'p': U12 @ Dz, 'w': -self.nu * _D, 'T': -U02 @ Id},
            'T': {'T': -self.kappa * _D},
        }
        self.setup_L(L_lhs)

        # mass matrix
        _U02 = U02 @ Id
        M_lhs = {i: {i: _U02} for i in ['u', 'v', 'w', 'T']}
        self.setup_M(M_lhs)

        # BCs
        self.add_BC(
            component='p', equation='p', axis=2, v=self.BCs['p_integral'], kind='integral', line=-1, scalar=True
        )
        self.add_BC(component='T', equation='T', axis=2, x=-1, v=self.BCs['T_bottom'], kind='Dirichlet', line=-1)
        self.add_BC(component='T', equation='T', axis=2, x=1, v=self.BCs['T_top'], kind='Dirichlet', line=-2)
        self.add_BC(component='w', equation='w', axis=2, x=1, v=self.BCs['w_top'], kind='Dirichlet', line=-1)
        self.add_BC(component='w', equation='w', axis=2, x=-1, v=self.BCs['w_bottom'], kind='Dirichlet', line=-2)
        self.remove_BC(component='w', equation='w', axis=2, x=-1, kind='Dirichlet', line=-2, scalar=True)
        for comp in ['u', 'v']:
            self.add_BC(
                component=comp, equation=comp, axis=2, v=self.BCs[f'{comp}_top'], x=1, kind='Dirichlet', line=-2
            )
            self.add_BC(
                component=comp,
                equation=comp,
                axis=2,
                v=self.BCs[f'{comp}_bottom'],
                x=-1,
                kind='Dirichlet',
                line=-1,
            )

        # eliminate Nyquist mode if needed
        if nx % 2 == 0:
            Nyquist_mode_index = self.axes[0].get_Nyquist_mode_index()
            for component in self.components:
                self.add_BC(
                    component=component, equation=component, axis=0, kind='Nyquist', line=int(Nyquist_mode_index), v=0
                )
        if ny % 2 == 0:
            Nyquist_mode_index = self.axes[0].get_Nyquist_mode_index()
            for component in self.components:
                self.add_BC(
                    component=component, equation=component, axis=1, kind='Nyquist', line=int(Nyquist_mode_index), v=0
                )
        self.setup_BCs()

        self.work_counters['rhs'] = WorkCounter()

    def eval_f(self, u, *args, **kwargs):
        f = self.f_init

        if self.spectral_space:
            u_hat = u.copy()
        else:
            u_hat = self.transform(u)

        f_impl_hat = self.u_init_forward

        iu, iv, iw, iT, ip = self.index(['u', 'v', 'w', 'T', 'p'])
        derivative_indices = [iu, iv, iw, iT]

        # evaluate implicit terms
        f_impl_hat = -(self.L @ u_hat.flatten()).reshape(u_hat.shape)
        for i in derivative_indices:
            f_impl_hat[i] = (self.S2 @ f_impl_hat[i].flatten()).reshape(f_impl_hat[i].shape)
        f_impl_hat[ip] = (self.S1 @ f_impl_hat[ip].flatten()).reshape(f_impl_hat[ip].shape)

        if self.spectral_space:
            self.xp.copyto(f.impl, f_impl_hat)
        else:
            f.impl[:] = self.itransform(f_impl_hat).real

        # -------------------------------------------
        # treat convection explicitly with dealiasing

        # start by computing derivatives
        padding = (self.dealiasing,) * self.ndim
        derivatives = []
        u_hat_flat = [u_hat[i].flatten() for i in derivative_indices]

        _D_u_hat = self.u_init_forward
        for D in [self.Dx, self.Dy, self.Dz]:
            _D_u_hat *= 0
            for i in derivative_indices:
                self.xp.copyto(_D_u_hat[i], (D @ u_hat_flat[i]).reshape(_D_u_hat[i].shape))
            derivatives.append(self.itransform(_D_u_hat, padding=padding).real)

        u_pad = self.itransform(u_hat, padding=padding).real

        fexpl_pad = self.xp.zeros_like(u_pad)
        for i in derivative_indices:
            for i_vel, iD in zip([iu, iv, iw], range(self.ndim)):
                fexpl_pad[i] -= u_pad[i_vel] * derivatives[iD][i]

        if self.spectral_space:
            self.xp.copyto(f.expl, self.transform(fexpl_pad, padding=padding))
        else:
            f.expl[:] = self.itransform(self.transform(fexpl_pad, padding=padding)).real

        self.work_counters['rhs']()
        return f

    def u_exact(self, t=0, noise_level=1e-3, seed=99):
        assert t == 0
        assert (
            self.BCs['v_top'] == self.BCs['v_bottom']
        ), 'Initial conditions are only implemented for zero velocity gradient'

        me = self.spectral.u_init
        iu, iw, iT, ip = self.index(['u', 'w', 'T', 'p'])

        # linear temperature gradient
        assert self.Lz == 1
        for comp in ['T', 'u', 'v', 'w']:
            a = self.BCs[f'{comp}_top'] - self.BCs[f'{comp}_bottom']
            b = self.BCs[f'{comp}_bottom']
            me[self.index(comp)] = a * self.Z + b

        # perturb slightly
        rng = self.xp.random.default_rng(seed=seed)

        noise = self.spectral.u_init
        noise[iT] = rng.random(size=me[iT].shape)

        me[iT] += noise[iT].real * noise_level * (self.Z - 1) * (self.Z + 1)

        if self.spectral_space:
            me_hat = self.spectral.u_init_forward
            me_hat[:] = self.transform(me)
            return me_hat
        else:
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
        Plot the solution.

        Parameters
        ----------
        u : dtype_u
            Solution to be plotted
        t : float
            Time to display at the top of the figure
        fig : matplotlib.pyplot.figure.Figure
            Figure with the same structure as a figure generated by `self.get_fig`. If none is supplied, a new figure will be generated.
        quantity : (str)
            quantity you want to plot

        Returns
        -------
        None
        """
        fig = self.get_fig() if fig is None else fig
        axs = fig.axes

        imV = axs[1].pcolormesh(self.X, self.Z, self.compute_vorticity(u).real)

        if self.spectral_space:
            u = self.itransform(u)

        imT = axs[0].pcolormesh(self.X, self.Z, u[self.index(quantity)].real)

        for i, label in zip([0, 1], [rf'${quantity}$', 'vorticity']):
            axs[i].set_aspect(1)
            axs[i].set_title(label)

        if t is not None:
            fig.suptitle(f't = {t:.2f}')
        axs[1].set_xlabel(r'$x$')
        axs[1].set_ylabel(r'$z$')
        fig.colorbar(imT, self.cax[0])
        fig.colorbar(imV, self.cax[1])


class RayleighBenard3DHeterogeneous(RayleighBenard3D):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # copy matrices we need on CPU
        if self.useGPU:
            for key in ['BC_line_zero_matrix', 'BCs']:  # TODO complete this list!
                setattr(self.spectral, key, getattr(self.spectral, key).get())
            for key in ['Pl', 'Pr', 'M']:  # TODO complete this list!
                setattr(self, key, getattr(self, key).get())

            self.L_CPU = self.L.get()
        else:
            self.L_CPU = self.L.copy()

        # delete matrices we do not need on GPU
        for key in []:  # TODO: complete list
            delattr(self, key)

    def solve_system(self, rhs, dt, u0=None, *args, skip_itransform=False, **kwargs):
        """
        Do an implicit Euler step to solve M u_t + Lu = rhs, with M the mass matrix and L the linear operator as setup by
        ``GenericSpectralLinear.setup_L`` and ``GenericSpectralLinear.setup_M``.

        The implicit Euler step is (M - dt L) u = M rhs. Note that M need not be invertible as long as (M + dt*L) is.
        This means solving with dt=0 to mimic explicit methods does not work for all problems, in particular simple DAEs.

        Note that by putting M rhs on the right hand side, this function can only solve algebraic conditions equal to
        zero. If you want something else, it should be easy to overload this function.
        """

        sp = self.spectral.sparse_lib

        if self.spectral_space:
            rhs_hat = rhs.copy()
            if u0 is not None:
                u0_hat = u0.copy().flatten()
            else:
                u0_hat = None
        else:
            rhs_hat = self.spectral.transform(rhs)
            if u0 is not None:
                u0_hat = self.spectral.transform(u0).flatten()
            else:
                u0_hat = None

        # apply inverse right preconditioner to initial guess
        if u0_hat is not None and 'direct' not in self.solver_type:
            if not hasattr(self, '_Pr_inv'):
                self._PR_inv = self.linalg.splu(self.Pr.astype(complex)).solve
            u0_hat[...] = self._PR_inv(u0_hat)

        rhs_hat = (self.M @ rhs_hat.flatten()).reshape(rhs_hat.shape)
        rhs_hat = self.spectral.put_BCs_in_rhs_hat(rhs_hat)
        rhs_hat = self.Pl @ rhs_hat.flatten()

        if dt not in self.cached_factorizations.keys() or not self.solver_type.lower() == 'cached_direct':
            A = self.M + dt * self.L_CPU
            A = self.Pl @ self.spectral.put_BCs_in_matrix(A) @ self.Pr
            A = self.spectral.sparse_lib.csc_matrix(A)

            # if A.shape[0] < 200e20:
            #     import matplotlib.pyplot as plt

            #     # M = self.spectral.put_BCs_in_matrix(self.L.copy())
            #     M = A  # self.L
            #     im = plt.spy(M)
            #     plt.show()

        if 'ilu' in self.solver_type.lower():
            if dt not in self.cached_factorizations.keys():
                if len(self.cached_factorizations) >= self.max_cached_factorizations:
                    to_evict = list(self.cached_factorizations.keys())[0]
                    self.cached_factorizations.pop(to_evict)
                    self.logger.debug(f'Evicted matrix factorization for {to_evict=:.6f} from cache')
                iLU = self.linalg.spilu(
                    A, **{**self.preconditioner_args, 'drop_tol': dt * self.preconditioner_args['drop_tol']}
                )
                self.cached_factorizations[dt] = self.linalg.LinearOperator(A.shape, iLU.solve)
                self.logger.debug(f'Cached incomplete LU factorization for {dt=:.6f}')
                self.work_counters['factorizations']()
            M = self.cached_factorizations[dt]
        else:
            M = None
        info = 0

        if self.solver_type.lower() == 'cached_direct':
            if dt not in self.cached_factorizations.keys():
                if len(self.cached_factorizations) >= self.max_cached_factorizations:
                    self.cached_factorizations.pop(list(self.cached_factorizations.keys())[0])
                    self.logger.debug(f'Evicted matrix factorization for {dt=:.6f} from cache')
                self.cached_factorizations[dt] = self.spectral.linalg.factorized(A)
                self.logger.debug(f'Cached matrix factorization for {dt=:.6f}')
                self.work_counters['factorizations']()

            _sol_hat = self.cached_factorizations[dt](rhs_hat)
            self.logger.debug(f'Used cached matrix factorization for {dt=:.6f}')

        elif self.solver_type.lower() == 'direct':
            _sol_hat = sp.linalg.spsolve(A, rhs_hat)
        elif 'gmres' in self.solver_type.lower():
            _sol_hat, _ = sp.linalg.gmres(
                A,
                rhs_hat,
                x0=u0_hat,
                **self.solver_args,
                callback=self.work_counters[self.solver_type],
                callback_type='pr_norm',
                M=M,
            )
        elif self.solver_type.lower() == 'cg':
            _sol_hat, info = sp.linalg.cg(
                A, rhs_hat, x0=u0_hat, **self.solver_args, callback=self.work_counters[self.solver_type]
            )
        elif 'bicgstab' in self.solver_type.lower():
            _sol_hat, info = self.linalg.bicgstab(
                A,
                rhs_hat,
                x0=u0_hat,
                **self.solver_args,
                callback=self.work_counters[self.solver_type],
                M=M,
            )
        else:
            raise NotImplementedError(f'Solver {self.solver_type=} not implemented in {type(self).__name__}!')

        if info != 0:
            self.logger.warn(f'{self.solver_type} not converged! {info=}')

        sol_hat = self.spectral.u_init_forward
        sol_hat[...] = (self.Pr @ _sol_hat).reshape(sol_hat.shape)

        if self.spectral_space:
            return sol_hat
        else:
            sol = self.spectral.u_init
            sol[:] = self.spectral.itransform(sol_hat).real

            if self.spectral.debug:
                self.spectral.check_BCs(sol)

            return sol
