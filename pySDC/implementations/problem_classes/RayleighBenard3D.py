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

        Omega = [0, Lx) x [0, Ly] x (0, Lz)
        T(z=+1) = 0
        T(z=-1) = Lz
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
        Rayleigh=1e6,
        nx=64,
        ny=64,
        nz=32,
        BCs=None,
        dealiasing=1.5,
        comm=None,
        Lz=1,
        Lx=4,
        Ly=4,
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
        iw, iT = self.index(['w', 'T'])
        zAxis = self.spectral.axes[-1]

        if self.spectral_space:
            u_hat = u.copy()
        else:
            u_hat = self.transform(u)

        DzT_hat = (self.Dz @ u_hat[iT].flatten()).reshape(u_hat[iT].shape)

        # compute wT with dealiasing
        padding = (self.dealiasing,) * self.ndim
        u_pad = self.itransform(u_hat, padding=padding).real
        _me = self.xp.zeros_like(u_pad)
        _me[0] = u_pad[iw] * u_pad[iT]
        wT_hat = self.transform(_me, padding=padding)[0]

        nusselt_hat = (wT_hat / self.kappa - DzT_hat) * self.axes[-1].L

        if not hasattr(self, '_zInt'):
            self._zInt = zAxis.get_integration_matrix()

        # get coefficients for evaluation on the boundary
        top = zAxis.get_BC(kind='Dirichlet', x=1)
        bot = zAxis.get_BC(kind='Dirichlet', x=-1)

        integral_V = 0
        if self.comm.rank == 0:

            integral_z = (self._zInt @ nusselt_hat[0, 0]).real
            integral_z[0] = zAxis.get_integration_constant(integral_z, axis=-1)
            integral_V = ((top - bot) * integral_z).sum() * self.axes[0].L * self.axes[1].L / self.nx / self.ny

        Nusselt_V = self.comm.bcast(integral_V / self.spectral.V, root=0)

        Nusselt_t = self.comm.bcast(self.xp.sum(nusselt_hat.real[0, 0, :] * top, axis=-1) / self.nx / self.ny, root=0)
        Nusselt_b = self.comm.bcast(self.xp.sum(nusselt_hat.real[0, 0] * bot / self.nx / self.ny, axis=-1), root=0)

        return {
            'V': Nusselt_V,
            't': Nusselt_t,
            'b': Nusselt_b,
        }

    def get_frequency_spectrum(self, u):
        """
        Compute the frequency spectrum of the velocities in x and y direction in the horizontal plane for every point in
        z. If the problem is well resolved, the coefficients will decay quickly with the wave number, and the reverse
        indicates that the resolution is too low.

        The returned spectrum has three dimensions. The first is for component (i.e. u or v), the second is for every
        point in z and the third is the energy in every wave number.

        Args:
            u: The solution you want to compute the spectrum of

        Returns:
            RayleighBenard3D.xp.ndarray: wave numbers
            RayleighBenard3D.xp.ndarray: spectrum
        """
        xp = self.xp
        indices = slice(0, 2)

        # transform the solution to be in frequency space in x and y, but real space in z
        if self.spectral_space:
            u_hat = self.itransform(u, axes=(-1,))
        else:
            u_hat = self.transform(
                u,
                axes=(
                    -3,
                    -2,
                ),
            )
        u_hat = self.spectral.redistribute(u_hat, axis=2, forward_output=False)

        # compute "energy density" as absolute square of the velocity modes
        energy = (u_hat[indices] * xp.conjugate(u_hat[indices])).real / (self.axes[0].N ** 2 * self.axes[1].N ** 2)

        # prepare wave numbers at which to compute the spectrum
        abs_kx = xp.abs(self.Kx[:, :, 0])
        abs_ky = xp.abs(self.Ky[:, :, 0])

        unique_k = xp.unique(xp.append(xp.unique(abs_kx), xp.unique(abs_ky)))
        n_k = len(unique_k)

        # compute local spectrum
        local_spectrum = self.xp.empty(shape=(2, energy.shape[3], n_k))
        for i, k in zip(range(n_k), unique_k):
            mask = xp.logical_or(abs_kx == k, abs_ky == k)
            local_spectrum[..., i] = xp.sum(energy[indices, mask, :], axis=1)

        # assemble global spectrum from local spectra
        k_all = self.comm.allgather(unique_k)
        unique_k_all = []
        for k in k_all:
            unique_k_all = xp.unique(xp.append(unique_k_all, xp.unique(k)))
        n_k_all = len(unique_k_all)

        spectra = self.comm.allgather(local_spectrum)
        spectrum = self.xp.zeros(shape=(2, self.axes[2].N, n_k_all))
        for ks, _spectrum in zip(k_all, spectra):
            ks = list(ks)
            unique_k_all = list(unique_k_all)
            for k in ks:
                index_global = unique_k_all.index(k)
                index_local = ks.index(k)
                spectrum[..., index_global] += _spectrum[..., index_local]

        return xp.array(unique_k_all), spectrum
