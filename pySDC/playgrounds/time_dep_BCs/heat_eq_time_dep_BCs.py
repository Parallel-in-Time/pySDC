import numpy as np

from pySDC.core.problem import Problem
from pySDC.implementations.datatype_classes.mesh import mesh
from pySDC.implementations.problem_classes.generic_spectral import GenericSpectralLinear


class GenericSpectralLinearTimeDepBCs(GenericSpectralLinear):
    def solve_system(self, rhs, dt, u0=None, t=0, *args, **kwargs):
        """
        Do an implicit Euler step to solve M u_t + Lu = rhs, with M the mass matrix and L the linear operator as setup by
        ``GenericSpectralLinear.setup_L`` and ``GenericSpectralLinear.setup_M``.

        The implicit Euler step is (M - dt L) u = M rhs. Note that M need not be invertible as long as (M + dt*L) is.
        This means solving with dt=0 to mimic explicit methods does not work for all problems, in particular simple DAEs.

        Note that by putting M rhs on the right hand side, this function can only solve algebraic conditions equal to
        zero. If you want something else, it should be easy to overload this function.
        """

        self.heterogeneous_setup()

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
        self.put_time_dep_BCs_in_rhs(
            rhs_hat, t
        )  # this line is the difference between this and the generic implementation
        rhs_hat = self.Pl @ rhs_hat.flatten()

        if dt not in self.cached_factorizations.keys():
            if self.heterogeneous:
                M = self.M_CPU
                L = self.L_CPU
                Pl = self.Pl_CPU
                Pr = self.Pr_CPU
            else:
                M = self.M
                L = self.L
                Pl = self.Pl
                Pr = self.Pr

            A = M + dt * L
            A = Pl @ self.spectral.put_BCs_in_matrix(A) @ Pr

        if dt not in self.cached_factorizations.keys():
            if len(self.cached_factorizations) >= self.max_cached_factorizations:
                self.cached_factorizations.pop(list(self.cached_factorizations.keys())[0])
                self.logger.debug(f'Evicted matrix factorization for {dt=:.6f} from cache')

            solver = self.spectral.linalg.factorized(A)

            self.cached_factorizations[dt] = solver
            self.logger.debug(f'Cached matrix factorization for {dt=:.6f}')
            self.work_counters['factorizations']()

        _sol_hat = self.cached_factorizations[dt](rhs_hat)
        self.work_counters[self.solver_type]()
        self.logger.debug(f'Used cached matrix factorization for {dt=:.6f}')

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


class Heat1DTimeDependentBCs(GenericSpectralLinearTimeDepBCs):
    """
    1D Heat equation with time-dependent Dirichlet Boundary conditions discretized on (-1, 1) using an ultraspherical spectral method.
    """

    dtype_u = mesh
    dtype_f = mesh

    def __init__(self, nvars=128, a=1, b=2, f=1, nu=1e-2, ft=np.pi, **kwargs):
        """
        Constructor. `kwargs` are forwarded to parent class constructor.

        Args:
            nvars (int): Resolution
            a (float): Left BC value at t=0
            b (float): Right BC value at t=0
            f (int): Frequency of the solution
            nu (float): Diffusion parameter
            ft (int): frequency of the BCs in time
        """
        self._makeAttributeAndRegister('nvars', 'a', 'b', 'f', 'nu', 'ft', localVars=locals(), readOnly=True)

        bases = [{'base': 'ultraspherical', 'N': nvars}]
        components = ['u']

        GenericSpectralLinear.__init__(self, bases, components, real_spectral_coefficients=True, **kwargs)

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

        if self.spectral_space:
            u_hat = u.copy()
        else:
            u_hat = self.transform(u)

        u_hat[iu] = (self.nu * (self.Dxx @ u_hat[iu].flatten())).reshape(u_hat[iu].shape)

        if self.spectral_space:
            me = u_hat
        else:
            me = self.itransform(u_hat).real

        f[iu][...] = me[iu]
        return f

    def u_exact(self, t=0):
        """
        Get initial conditions

        Args:
            t (float): When you want the exact solution

        Returns:
            Heat1DUltraspherical.dtype_u: Exact solution
        """
        assert t == 0

        xp = self.xp
        iu = self.index('u')
        u = self.spectral.u_init_physical

        u[iu] = (
            xp.sin(np.pi * self.x) * xp.exp(-self.nu * (self.f * np.pi) ** 2 * t)
            + (self.b - self.a) / 2 * self.x
            + (self.b + self.a) / 2
        )

        if self.spectral_space:
            u_hat = self.spectral.u_init_forward
            u_hat[...] = self.transform(u)
            u = u_hat

        # apply BCs
        u = self.solve_system(u, 1e-9, u, t)

        return u

    def put_time_dep_BCs_in_rhs(self, rhs_hat, t):
        """
        Put the time dependent BCs in the right hand side.

        In this simple 1D case the BCs are simply in the last two lines of the problem, so we can put there whatever we want.
        Note that in 2D you essentially do the same, but you need to unflatten the RHS, put the BCs in the last lines, and then reflatten.
        """
        rhs_hat[0, -1] = self.a * self.xp.cos(t * self.ft)
        rhs_hat[0, -2] = self.b * self.xp.cos(t * self.ft)
        return rhs_hat

    def get_fig(self):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        return fig

    def plot(self, u, t, fig):
        if self.spectral_space:
            u = self.itransform(u)
        ax = fig.get_axes()[0]
        ax.cla()
        ax.plot(self.x, u[0])
