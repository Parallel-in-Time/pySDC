import numpy as np
from scipy import sparse as sp

from pySDC.core.problem import Problem
from pySDC.implementations.datatype_classes.mesh import mesh
from pySDC.implementations.problem_classes.generic_spectral import GenericSpectralLinear


class Heat1DChebychov(GenericSpectralLinear):
    dtype_u = mesh
    dtype_f = mesh

    def __init__(self, nvars=128, a=0, b=0, f=1, nu=1.0, mode='T2U', **kwargs):
        self._makeAttributeAndRegister(*locals().keys(), localVars=locals(), readOnly=True)

        bases = [{'base': 'chebychov', 'N': nvars}]
        components = ['u', 'ux']

        super().__init__(bases, components, **kwargs)

        self.x = self.get_grid()[0]

        I = self.get_Id()
        Dx = self.get_differentiation_matrix(axes=(0,))
        self.Dx = Dx

        self.T2U = self.get_basis_change_matrix(conv=mode)

        L_lhs = {
            'ux': {'u': -self.T2U @ Dx, 'ux': self.T2U @ I},
            'u': {'ux': -nu * (self.T2U @ Dx)},
        }
        self.setup_L(L_lhs)

        M_lhs = {'u': {'u': self.T2U @ I}}
        self.setup_M(M_lhs)

        self.add_BC(component='u', equation='u', axis=0, x=-1, v=a, kind="Dirichlet")
        self.add_BC(component='u', equation='ux', axis=0, x=1, v=b, kind="Dirichlet")
        self.setup_BCs()

    def eval_f(self, u, *args, **kwargs):
        f = self.f_init
        iu, iux = self.index(self.components)

        me_hat = self.u_init_forward
        me_hat[:] = self.transform(u)
        me_hat[iu] = (self.nu * self.Dx @ me_hat[iux].flatten()).reshape(me_hat[iu].shape)
        me = self.itransform(me_hat).real

        f[iu] = me[iu]
        return f

    def u_exact(self, t, noise=0):
        xp = self.xp
        iu, iux = self.index(self.components)
        u = self.u_init

        u[iu] = (
            xp.sin(self.f * np.pi * self.x) * xp.exp(-self.nu * (self.f * np.pi) ** 2 * t)
            + (self.b - self.a) / 2 * self.x
            + (self.b + self.a) / 2
        )
        u[iux] = (
            self.f * np.pi * xp.cos(self.f * np.pi * self.x) * xp.exp(-self.nu * (self.f * np.pi) ** 2 * t)
            + (self.b - self.a) / 2
        )

        if noise > 0:
            assert t == 0
            _noise = self.u_init
            rng = self.xp.random.default_rng(seed=666)
            _noise[iu] = rng.normal(size=u[iu].shape)
            noise_hat = self.transform(_noise)
            low_pass = self.get_filter_matrix(axis=0, kmax=self.nvars - 2)
            noise_hat[iu] = (low_pass @ noise_hat[iu].flatten()).reshape(noise_hat[iu].shape)
            _noise[:] = self.itransform(noise_hat)
            u += _noise * noise * (self.x - 1) * (self.x + 1)

        self.check_BCs(u)
        return u


class Heat1DUltraspherical(GenericSpectralLinear):
    dtype_u = mesh
    dtype_f = mesh

    def __init__(self, nvars=128, a=0, b=0, f=1, nu=1.0, **kwargs):
        self._makeAttributeAndRegister(*locals().keys(), localVars=locals(), readOnly=True)

        bases = [{'base': 'ultraspherical', 'N': nvars}]
        components = ['u']

        GenericSpectralLinear.__init__(self, bases, components, **kwargs)

        self.x = self.get_grid()[0]

        I = self.get_Id()
        Dxx = self.get_differentiation_matrix(axes=(0,), p=2)

        S2 = self.get_basis_change_matrix(p_in=2, p_out=0)
        U2 = self.get_basis_change_matrix(p_in=0, p_out=2)

        self.Dxx = S2 @ Dxx

        L_lhs = {
            'u': {'u': -nu * Dxx},
        }
        self.setup_L(L_lhs)

        M_lhs = {'u': {'u': U2 @ I}}
        self.setup_M(M_lhs)

        self.add_BC(component='u', equation='u', axis=0, x=-1, v=a, kind="Dirichlet", line=-1)
        self.add_BC(component='u', equation='u', axis=0, x=1, v=b, kind="Dirichlet", line=-2)
        self.setup_BCs()

    def eval_f(self, u, *args, **kwargs):
        f = self.f_init
        iu = self.index('u')

        me_hat = self.u_init_forward
        me_hat[:] = self.transform(u)
        me_hat[iu] = (self.nu * (self.Dxx @ me_hat[iu].flatten())).reshape(me_hat[iu].shape)
        me = self.itransform(me_hat).real

        f[iu][...] = me[iu]
        return f

    def u_exact(self, t, noise=0):
        xp = self.xp
        iu = self.index('u')
        u = self.u_init

        u[iu] = (
            xp.sin(self.f * np.pi * self.x) * xp.exp(-self.nu * (self.f * np.pi) ** 2 * t)
            + (self.b - self.a) / 2 * self.x
            + (self.b + self.a) / 2
        )

        if noise > 0:
            assert t == 0
            _noise = self.u_init
            rng = self.xp.random.default_rng(seed=666)
            _noise[iu] = rng.normal(size=u[iu].shape)
            noise_hat = self.transform(_noise)
            low_pass = self.get_filter_matrix(axis=0, kmax=self.nvars - 2)
            noise_hat[iu] = (low_pass @ noise_hat[iu].flatten()).reshape(noise_hat[iu].shape)
            _noise[:] = self.itransform(noise_hat)
            u += _noise * noise * (self.x - 1) * (self.x + 1)

        self.check_BCs(u)
        return u


class Heat2DChebychov(GenericSpectralLinear):
    dtype_u = mesh
    dtype_f = mesh

    def __init__(self, nx=128, ny=128, base_x='fft', base_y='chebychov', a=0, b=0, c=0, fx=1, fy=1, nu=1.0, **kwargs):
        assert nx % 2 == 1 or base_x == 'fft'
        assert ny % 2 == 1 or base_y == 'fft'
        self._makeAttributeAndRegister(*locals().keys(), localVars=locals(), readOnly=True)

        bases = [{'base': base_x, 'N': nx}, {'base': base_y, 'N': ny}]
        components = ['u', 'ux', 'uy']

        super().__init__(bases, components, Dirichlet_recombination=False, **kwargs)

        self.Y, self.X = self.get_grid()

        I = self.get_Id()
        self.Dx = self.get_differentiation_matrix(axes=(0,))
        self.Dy = self.get_differentiation_matrix(axes=(1,))

        L_lhs = {
            'ux': {'u': -self.Dx, 'ux': I},
            'uy': {'u': -self.Dy, 'uy': I},
            'u': {'ux': -nu * self.Dx, 'uy': -nu * self.Dy},
        }
        self.setup_L(L_lhs)

        M_lhs = {'u': {'u': I}}
        self.setup_M(M_lhs)

        for base in [base_x, base_y]:
            assert base in ['chebychov', 'fft']

        alpha = (self.b - self.a) / 2.0
        beta = (self.c - self.b) / 2.0
        gamma = (self.c + self.a) / 2.0

        if base_x == 'chebychov':
            y = self.Y[0, :]
            if self.base_y == 'fft':
                self.add_BC(component='u', equation='u', axis=0, x=-1, v=beta * y - alpha + gamma, kind='Dirichlet')
            self.add_BC(component='ux', equation='ux', axis=0, v=2 * alpha, kind='integral')
        else:
            assert a == b, f'Need periodic boundary conditions in x for {base_x} method!'
        if base_y == 'chebychov':
            x = self.X[:, 0]
            self.add_BC(component='u', equation='u', axis=1, x=-1, v=alpha * x - beta + gamma, kind='Dirichlet')
            self.add_BC(component='uy', equation='uy', axis=1, v=2 * beta, kind='integral')
        else:
            assert c == b, f'Need periodic boundary conditions in y for {base_y} method!'
        self.setup_BCs()

    def eval_f(self, u, *args, **kwargs):
        f = self.f_init
        iu, iux, iuy = self.index(self.components)

        me_hat = self.u_init_forward
        me_hat[:] = self.transform(u)
        me_hat[iu] = self.nu * (self.Dx @ me_hat[iux].flatten() + self.Dy @ me_hat[iuy].flatten()).reshape(
            me_hat[iu].shape
        )
        me = self.itransform(me_hat)

        f[self.index("u")] = me[iu].real
        return f

    def u_exact(self, t):
        xp = self.xp
        iu, iux, iuy = self.index(self.components)
        u = self.u_init

        fx = self.fx if self.base_x == 'fft' else np.pi * self.fx
        fy = self.fy if self.base_y == 'fft' else np.pi * self.fy

        time_dep = xp.exp(-self.nu * (fx**2 + fy**2) * t)

        alpha = (self.b - self.a) / 2.0
        beta = (self.c - self.b) / 2.0
        gamma = (self.c + self.a) / 2.0

        u[iu] = xp.sin(fx * self.X) * xp.sin(fy * self.Y) * time_dep + alpha * self.X + beta * self.Y + gamma
        u[iux] = fx * xp.cos(fx * self.X) * xp.sin(fy * self.Y) * time_dep + alpha
        u[iuy] = fy * xp.sin(fx * self.X) * xp.cos(fy * self.Y) * time_dep + beta

        return u


class Heat2DUltraspherical(GenericSpectralLinear):
    dtype_u = mesh
    dtype_f = mesh

    def __init__(
        self, nx=128, ny=128, base_x='fft', base_y='ultraspherical', a=0, b=0, c=0, fx=1, fy=1, nu=1.0, **kwargs
    ):
        self._makeAttributeAndRegister(*locals().keys(), localVars=locals(), readOnly=True)

        bases = [{'base': base_x, 'N': nx}, {'base': base_y, 'N': ny}]
        components = ['u']

        super().__init__(bases, components, Dirichlet_recombination=False, **kwargs)

        self.Y, self.X = self.get_grid()

        self.Dxx = self.get_differentiation_matrix(axes=(0,), p=2)
        self.Dyy = self.get_differentiation_matrix(axes=(1,), p=2)
        self.S2 = self.get_basis_change_matrix(p=2)
        I = self.get_Id()

        L_lhs = {
            'u': {'u': -nu * self.Dxx - nu * self.Dyy},
        }
        self.setup_L(L_lhs)

        M_lhs = {'u': {'u': I}}
        self.setup_M(M_lhs)

        for base in [base_x, base_y]:
            assert base in ['ultraspherical', 'fft']

        alpha = (self.b - self.a) / 2.0
        beta = (self.c - self.b) / 2.0
        gamma = (self.c + self.a) / 2.0

        if base_x == 'ultraspherical':
            y = self.Y[0, :]
            if self.base_y == 'fft':
                self.add_BC(component='u', equation='u', axis=0, x=-1, v=beta * y - alpha + gamma, kind='Dirichlet')
            self.add_BC(component='u', equation='u', axis=0, v=beta * y + alpha + gamma, x=1, line=-2, kind='Dirichlet')
        else:
            assert a == b, f'Need periodic boundary conditions in x for {base_x} method!'
        if base_y == 'ultraspherical':
            x = self.X[:, 0]
            self.add_BC(
                component='u', equation='u', axis=1, x=-1, v=alpha * x - beta + gamma, kind='Dirichlet', line=-1
            )
            self.add_BC(
                component='u', equation='u', axis=1, x=+1, v=alpha * x + beta + gamma, kind='Dirichlet', line=-2
            )
        else:
            assert c == b, f'Need periodic boundary conditions in y for {base_y} method!'
        self.setup_BCs()

    def eval_f(self, u, *args, **kwargs):
        f = self.f_init
        iu = self.index('u')

        me_hat = self.u_init_forward
        me_hat[:] = self.transform(u)
        me_hat[iu] = self.nu * (self.S2 @ (self.Dxx + self.Dyy) @ me_hat[iu].flatten()).reshape(me_hat[iu].shape)
        me = self.itransform(me_hat)

        f[iu] = me[iu].real
        return f

    def u_exact(self, t):
        xp = self.xp
        iu = self.index('u')
        u = self.u_init

        fx = self.fx if self.base_x == 'fft' else np.pi * self.fx
        fy = self.fy if self.base_y == 'fft' else np.pi * self.fy

        time_dep = xp.exp(-self.nu * (fx**2 + fy**2) * t)

        alpha = (self.b - self.a) / 2.0
        beta = (self.c - self.b) / 2.0
        gamma = (self.c + self.a) / 2.0

        u[iu] = xp.sin(fx * self.X) * xp.sin(fy * self.Y) * time_dep + alpha * self.X + beta * self.Y + gamma

        return u
