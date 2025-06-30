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
    Rayleigh-Benard Convection is a variation of incompressible fluid dynamics.

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

        Omega = [0, Lx) x [0, Ly) x (0, Lz)
        T(z=Lz) = 0
        T(z=0) = Lz
        u(z=0) = v(z=0) = w(z=0) = 0
        u(z=Lz) = v(z=Lz) = w(z=Lz) = 0
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

        # compute rescaled Rayleigh number to extract viscosity and thermal diffusivity
        Ra = Rayleigh / (max([abs(BCs['T_top'] - BCs['T_bottom']), np.finfo(float).eps]) * self.axes[2].L ** 3)
        self.kappa = (Ra * Prandtl) ** (-1 / 2.0)
        self.nu = (Ra / Prandtl) ** (-1 / 2.0)

        # construct operators
        L_lhs = {
            'p': {'u': U01 @ Dx, 'v': U01 @ Dy, 'w': Dz},  # divergence free constraint
            'u': {'p': U02 @ Dx, 'u': -self.nu * (U02 @ (Dxx + Dyy) + Dzz)},
            'v': {'p': U02 @ Dy, 'v': -self.nu * (U02 @ (Dxx + Dyy) + Dzz)},
            'w': {'p': U12 @ Dz, 'w': -self.nu * (U02 @ (Dxx + Dyy) + Dzz), 'T': -U02 @ Id},
            'T': {'T': -self.kappa * (U02 @ (Dxx + Dyy) + Dzz)},
        }
        self.setup_L(L_lhs)

        # mass matrix
        M_lhs = {i: {i: U02 @ Id} for i in ['u', 'v', 'w', 'T']}
        self.setup_M(M_lhs)

        # Prepare going from second (first for divergence free equation) derivative basis back to Chebychev-T
        self.base_change = self._setup_operator({**{comp: {comp: S2} for comp in ['u', 'v', 'w', 'T']}, 'p': {'p': S1}})

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

        # evaluate implicit terms
        if not hasattr(self, '_L_T_base'):
            self._L_T_base = self.base_change @ self.L
        f_impl_hat = -(self._L_T_base @ u_hat.flatten()).reshape(u_hat.shape)

        if self.spectral_space:
            f.impl[:] = f_impl_hat
        else:
            f.impl[:] = self.itransform(f_impl_hat).real

        # -------------------------------------------
        # treat convection explicitly with dealiasing

        # start by computing derivatives
        if not hasattr(self, '_Dx_expanded') or not hasattr(self, '_Dz_expanded'):
            Dz = self.Dz
            Dy = self.Dy
            Dx = self.Dx

            self._Dx_expanded = self._setup_operator(
                {'u': {'u': Dx}, 'v': {'v': Dx}, 'w': {'w': Dx}, 'T': {'T': Dx}, 'p': {}}
            )
            self._Dy_expanded = self._setup_operator(
                {'u': {'u': Dy}, 'v': {'v': Dy}, 'w': {'w': Dy}, 'T': {'T': Dy}, 'p': {}}
            )
            self._Dz_expanded = self._setup_operator(
                {'u': {'u': Dz}, 'v': {'v': Dz}, 'w': {'w': Dz}, 'T': {'T': Dz}, 'p': {}}
            )
        Dx_u_hat = (self._Dx_expanded @ u_hat.flatten()).reshape(u_hat.shape)
        Dy_u_hat = (self._Dy_expanded @ u_hat.flatten()).reshape(u_hat.shape)
        Dz_u_hat = (self._Dz_expanded @ u_hat.flatten()).reshape(u_hat.shape)

        padding = (self.dealiasing,) * self.ndim
        Dx_u_pad = self.itransform(Dx_u_hat, padding=padding).real
        Dy_u_pad = self.itransform(Dy_u_hat, padding=padding).real
        Dz_u_pad = self.itransform(Dz_u_hat, padding=padding).real
        u_pad = self.itransform(u_hat, padding=padding).real

        fexpl_pad = self.xp.zeros_like(u_pad)
        fexpl_pad[iu][:] = -(u_pad[iu] * Dx_u_pad[iu] + u_pad[iv] * Dy_u_pad[iu] + u_pad[iw] * Dz_u_pad[iu])
        fexpl_pad[iv][:] = -(u_pad[iu] * Dx_u_pad[iv] + u_pad[iv] * Dy_u_pad[iv] + u_pad[iw] * Dz_u_pad[iv])
        fexpl_pad[iw][:] = -(u_pad[iu] * Dx_u_pad[iw] + u_pad[iv] * Dy_u_pad[iw] + u_pad[iw] * Dz_u_pad[iw])
        fexpl_pad[iT][:] = -(u_pad[iu] * Dx_u_pad[iT] + u_pad[iv] * Dy_u_pad[iT] + u_pad[iw] * Dz_u_pad[iT])

        if self.spectral_space:
            f.expl[:] = self.transform(fexpl_pad, padding=padding)
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
