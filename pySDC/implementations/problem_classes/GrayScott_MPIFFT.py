import scipy.sparse as sp
from mpi4py import MPI
from mpi4py_fft import PFFT

from pySDC.core.errors import ProblemError
from pySDC.core.problem import Problem, WorkCounter
from pySDC.implementations.datatype_classes.mesh import mesh, imex_mesh, comp2_mesh
from pySDC.implementations.problem_classes.generic_MPIFFT_Laplacian import IMEX_Laplacian_MPIFFT

from mpi4py_fft import newDistArray


class grayscott_imex_diffusion(IMEX_Laplacian_MPIFFT):
    r"""
    The Gray-Scott system [1]_ describes a reaction-diffusion process of two substances :math:`u` and :math:`v`,
    where they diffuse over time. During the reaction :math:`u` is used up with overall decay rate :math:`B`,
    whereas :math:`v` is produced with feed rate :math:`A`. :math:`D_u,\, D_v` are the diffusion rates for
    :math:`u,\, v`. Here, the process is described by the :math:`N`-dimensional model

    .. math::
        \frac{\partial u}{\partial t} = D_u \Delta u - u v^2 + A (1 - u),

    .. math::
        \frac{\partial v}{\partial t} = D_v \Delta v + u v^2 - B u

    in :math:`x \in \Omega:=[-L/2, L/2]^N` with :math:`N=2,3`. Spatial discretization is done by using
    Fast Fourier transformation for solving the linear parts provided by ``mpi4py-fft`` [2]_, see also
    https://mpi4py-fft.readthedocs.io/en/latest/.

    This class implements the problem for *semi-explicit* time-stepping (diffusion is treated implicitly, and reaction
    is computed in explicit fashion).

    Parameters
    ----------
    nvars : tuple of int, optional
        Spatial resolution, i.e., number of degrees of freedom in space. Should be a tuple, e.g. ``nvars=(127, 127)``.
    Du : float, optional
        Diffusion rate for :math:`u`.
    Dv: float, optional
        Diffusion rate for :math:`v`.
    A : float, optional
        Feed rate for :math:`v`.
    B : float, optional
        Overall decay rate for :math:`u`.
    spectral : bool, optional
        If True, the solution is computed in spectral space.
    L : int, optional
        Denotes the period of the function to be approximated for the Fourier transform.
    comm : COMM_WORLD, optional
        Communicator for ``mpi4py-fft``.

    Attributes
    ----------
    fft : PFFT
        Object for parallel FFT transforms.
    X : mesh-grid
        Grid coordinates in real space.
    ndim : int
        Number of spatial dimensions.
    Ku : matrix
        Laplace operator in spectral space (u component).
    Kv : matrix
        Laplace operator in spectral space (v component).

    References
    ----------
    .. [1] Autocatalytic reactions in the isothermal, continuous stirred tank reactor: Isolas and other forms
        of multistability. P. Gray, S. K. Scott. Chem. Eng. Sci. 38, 1 (1983).
    .. [2] Lisandro Dalcin, Mikael Mortensen, David E. Keyes. Fast parallel multidimensional FFT using advanced MPI.
        Journal of Parallel and Distributed Computing (2019).
    .. [3] https://www.chebfun.org/examples/pde/GrayScott.html
    """

    def __init__(self, Du=1.0, Dv=0.01, A=0.09, B=0.086, **kwargs):
        kwargs['L'] = 2.0
        super().__init__(dtype='d', alpha=1.0, x0=-kwargs['L'] / 2.0, **kwargs)

        # prepare the array with two components
        shape = (2,) + (self.init[0])
        self.iU = 0
        self.iV = 1
        self.ncomp = 2  # needed for transfer class
        self.init = (shape, self.comm, self.xp.dtype('float'))

        self._makeAttributeAndRegister('Du', 'Dv', 'A', 'B', localVars=locals(), readOnly=True)

        # prepare "Laplacians"
        self.Ku = -self.Du * self.K2
        self.Kv = -self.Dv * self.K2

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
            f.impl[0, ...] = self.Ku * u[0, ...]
            f.impl[1, ...] = self.Kv * u[1, ...]
            tmpu = newDistArray(self.fft, False)
            tmpv = newDistArray(self.fft, False)
            tmpu[:] = self.fft.backward(u[0, ...], tmpu)
            tmpv[:] = self.fft.backward(u[1, ...], tmpv)
            tmpfu = -tmpu * tmpv**2 + self.A * (1 - tmpu)
            tmpfv = tmpu * tmpv**2 - self.B * tmpv
            f.expl[0, ...] = self.fft.forward(tmpfu)
            f.expl[1, ...] = self.fft.forward(tmpfv)

        else:
            u_hat = self.fft.forward(u[0, ...])
            lap_u_hat = self.Ku * u_hat
            f.impl[0, ...] = self.fft.backward(lap_u_hat, f.impl[0, ...])
            u_hat = self.fft.forward(u[1, ...])
            lap_u_hat = self.Kv * u_hat
            f.impl[1, ...] = self.fft.backward(lap_u_hat, f.impl[1, ...])
            f.expl[0, ...] = -u[0, ...] * u[1, ...] ** 2 + self.A * (1 - u[0, ...])
            f.expl[1, ...] = u[0, ...] * u[1, ...] ** 2 - self.B * u[1, ...]

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
        if self.spectral:
            me[0, ...] = rhs[0, ...] / (1.0 - factor * self.Ku)
            me[1, ...] = rhs[1, ...] / (1.0 - factor * self.Kv)

        else:
            rhs_hat = self.fft.forward(rhs[0, ...])
            rhs_hat /= 1.0 - factor * self.Ku
            me[0, ...] = self.fft.backward(rhs_hat, me[0, ...])
            rhs_hat = self.fft.forward(rhs[1, ...])
            rhs_hat /= 1.0 - factor * self.Kv
            me[1, ...] = self.fft.backward(rhs_hat, me[1, ...])

        return me

    def u_exact(self, t):
        r"""
        Routine to compute the exact solution at time :math:`t = 0`, see [3]_.

        Parameters
        ----------
        t : float
            Time of the exact solution.

        Returns
        -------
        me : dtype_u
            Exact solution.
        """
        assert t == 0.0, 'Exact solution only valid as initial condition'
        assert self.ndim == 2, 'The initial conditions are 2D for now..'

        me = self.dtype_u(self.init, val=0.0)

        # This assumes that the box is [-L/2, L/2]^2
        if self.spectral:
            tmp = 1.0 - self.xp.exp(-80.0 * ((self.X[0] + 0.05) ** 2 + (self.X[1] + 0.02) ** 2))
            me[0, ...] = self.fft.forward(tmp)
            tmp = self.xp.exp(-80.0 * ((self.X[0] - 0.05) ** 2 + (self.X[1] - 0.02) ** 2))
            me[1, ...] = self.fft.forward(tmp)
        else:
            me[0, ...] = 1.0 - self.xp.exp(-80.0 * ((self.X[0] + 0.05) ** 2 + (self.X[1] + 0.02) ** 2))
            me[1, ...] = self.xp.exp(-80.0 * ((self.X[0] - 0.05) ** 2 + (self.X[1] - 0.02) ** 2))

        return me


class grayscott_imex_linear(grayscott_imex_diffusion):
    r"""
    The Gray-Scott system [1]_ describes a reaction-diffusion process of two substances :math:`u` and :math:`v`,
    where they diffuse over time. During the reaction :math:`u` is used up with overall decay rate :math:`B`,
    whereas :math:`v` is produced with feed rate :math:`A`. :math:`D_u,\, D_v` are the diffusion rates for
    :math:`u,\, v`. The model with linear (reaction) part is described by the :math:`N`-dimensional model

    .. math::
        \frac{d u}{d t} = D_u \Delta u - u v^2 + A,

    .. math::
        \frac{d v}{d t} = D_v \Delta v + u v^2

    in :math:`x \in \Omega:=[-L/2, L/2]^N` with :math:`N=2,3`. Spatial discretization is done by using
    Fast Fourier transformation for solving the linear parts provided by ``mpi4py-fft`` [2]_, see also
    https://mpi4py-fft.readthedocs.io/en/latest/.

    This class implements the problem for *semi-explicit* time-stepping (diffusion is treated implicitly, and linear
    part is computed in an explicit way).
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
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
            f.impl[0, ...] = self.Ku * u[0, ...]
            f.impl[1, ...] = self.Kv * u[1, ...]
            tmpu = newDistArray(self.fft, False)
            tmpv = newDistArray(self.fft, False)
            tmpu[:] = self.fft.backward(u[0, ...], tmpu)
            tmpv[:] = self.fft.backward(u[1, ...], tmpv)
            tmpfu = -tmpu * tmpv**2 + self.A
            tmpfv = tmpu * tmpv**2
            f.expl[0, ...] = self.fft.forward(tmpfu)
            f.expl[1, ...] = self.fft.forward(tmpfv)

        else:
            u_hat = self.fft.forward(u[0, ...])
            lap_u_hat = self.Ku * u_hat
            f.impl[0, ...] = self.fft.backward(lap_u_hat, f.impl[0, ...])
            u_hat = self.fft.forward(u[1, ...])
            lap_u_hat = self.Kv * u_hat
            f.impl[1, ...] = self.fft.backward(lap_u_hat, f.impl[1, ...])
            f.expl[0, ...] = -u[0, ...] * u[1, ...] ** 2 + self.A
            f.expl[1, ...] = u[0, ...] * u[1, ...] ** 2

        self.work_counters['rhs']()
        return f


class grayscott_mi_diffusion(grayscott_imex_diffusion):
    r"""
    The Gray-Scott system [1]_ describes a reaction-diffusion process of two substances :math:`u` and :math:`v`,
    where they diffuse over time. During the reaction :math:`u` is used up with overall decay rate :math:`B`,
    whereas :math:`v` is produced with feed rate :math:`A`. :math:`D_u,\, D_v` are the diffusion rates for
    :math:`u,\, v`. Here, the process is described by the :math:`N`-dimensional model

    .. math::
        \frac{\partial u}{\partial t} = D_u \Delta u - u v^2 + A (1 - u),

    .. math::
        \frac{\partial v}{\partial t} = D_v \Delta v + u v^2 - B u

    in :math:`x \in \Omega:=[-L/2, L/2]^N` with :math:`N=2,3`. Spatial discretization is done by using
    Fast Fourier transformation for solving the linear parts provided by ``mpi4py-fft`` [2]_, see also
    https://mpi4py-fft.readthedocs.io/en/latest/.

    This class implements the problem for *multi-implicit* time-stepping, i.e., both diffusion and reaction part will be treated
    implicitly.

    Parameters
    ----------
    nvars : tuple of int, optional
        Spatial resolution, i.e., number of degrees of freedom in space. Should be a tuple, e.g. ``nvars=(127, 127)``.
    Du : float, optional
        Diffusion rate for :math:`u`.
    Dv: float, optional
        Diffusion rate for :math:`v`.
    A : float, optional
        Feed rate for :math:`v`.
    B : float, optional
        Overall decay rate for :math:`u`.
    spectral : bool, optional
        If True, the solution is computed in spectral space.
    L : int, optional
        Denotes the period of the function to be approximated for the Fourier transform.
    comm : COMM_WORLD, optional
        Communicator for ``mpi4py-fft``.

    Attributes
    ----------
    fft : PFFT
        Object for parallel FFT transforms.
    X : mesh-grid
        Grid coordinates in real space.
    ndim : int
        Number of spatial dimensions.
    Ku : matrix
        Laplace operator in spectral space (u component).
    Kv : matrix
        Laplace operator in spectral space (v component).

    References
    ----------
    .. [1] Autocatalytic reactions in the isothermal, continuous stirred tank reactor: Isolas and other forms
        of multistability. P. Gray, S. K. Scott. Chem. Eng. Sci. 38, 1 (1983).
    .. [2] Lisandro Dalcin, Mikael Mortensen, David E. Keyes. Fast parallel multidimensional FFT using advanced MPI.
        Journal of Parallel and Distributed Computing (2019).
    """

    dtype_f = comp2_mesh

    def __init__(
        self,
        newton_maxiter=100,
        newton_tol=1e-12,
        **kwargs,
    ):
        """Initialization routine"""
        super().__init__(**kwargs)
        # This may not run in parallel yet..
        assert self.comm.Get_size() == 1
        self.work_counters['newton'] = WorkCounter()
        self.Ku = -self.Du * self.K2
        self.Kv = -self.Dv * self.K2
        self._makeAttributeAndRegister('newton_maxiter', 'newton_tol', localVars=locals(), readOnly=False)

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
            f.comp1[0, ...] = self.Ku * u[0, ...]
            f.comp1[1, ...] = self.Kv * u[1, ...]
            tmpu = newDistArray(self.fft, False)
            tmpv = newDistArray(self.fft, False)
            tmpu[:] = self.fft.backward(u[0, ...], tmpu)
            tmpv[:] = self.fft.backward(u[1, ...], tmpv)
            tmpfu = -tmpu * tmpv**2 + self.A * (1 - tmpu)
            tmpfv = tmpu * tmpv**2 - self.B * tmpv
            f.comp2[0, ...] = self.fft.forward(tmpfu)
            f.comp2[1, ...] = self.fft.forward(tmpfv)

        else:
            u_hat = self.fft.forward(u[0, ...])
            lap_u_hat = self.Ku * u_hat
            f.comp1[0, ...] = self.fft.backward(lap_u_hat, f.comp1[0, ...])
            u_hat = self.fft.forward(u[1, ...])
            lap_u_hat = self.Kv * u_hat
            f.comp1[1, ...] = self.fft.backward(lap_u_hat, f.comp1[1, ...])
            f.comp2[0, ...] = -u[0, ...] * u[1, ...] ** 2 + self.A * (1 - u[0, ...])
            f.comp2[1, ...] = u[0, ...] * u[1, ...] ** 2 - self.B * u[1, ...]

        self.work_counters['rhs']()
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

        me = super().solve_system(rhs, factor, u0, t)
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
            tmpu[:] = self.fft.backward(u[0, ...], tmpu)
            tmpv[:] = self.fft.backward(u[1, ...], tmpv)
            tmprhsu = newDistArray(self.fft, False)
            tmprhsv = newDistArray(self.fft, False)
            tmprhsu[:] = self.fft.backward(rhs[0, ...], tmprhsu)
            tmprhsv[:] = self.fft.backward(rhs[1, ...], tmprhsv)

        else:
            tmpu = u[0, ...]
            tmpv = u[1, ...]
            tmprhsu = rhs[0, ...]
            tmprhsv = rhs[1, ...]

        # start newton iteration
        n = 0
        res = 99
        while n < self.newton_maxiter:
            # print(n, res)
            # form the function g with g(u) = 0
            tmpgu = tmpu - tmprhsu - factor * (-tmpu * tmpv**2 + self.A * (1 - tmpu))
            tmpgv = tmpv - tmprhsv - factor * (tmpu * tmpv**2 - self.B * tmpv)

            # if g is close to 0, then we are done
            res = max(self.xp.linalg.norm(tmpgu, self.xp.inf), self.xp.linalg.norm(tmpgv, self.xp.inf))
            if res < self.newton_tol:
                break

            # assemble dg
            dg00 = 1 - factor * (-(tmpv**2) - self.A)
            dg01 = -factor * (-2 * tmpu * tmpv)
            dg10 = -factor * (tmpv**2)
            dg11 = 1 - factor * (2 * tmpu * tmpv - self.B)

            # interleave and unravel to put into sparse matrix
            dg00I = self.xp.ravel(self.xp.kron(dg00, self.xp.array([1, 0])))
            dg01I = self.xp.ravel(self.xp.kron(dg01, self.xp.array([1, 0])))
            dg10I = self.xp.ravel(self.xp.kron(dg10, self.xp.array([1, 0])))
            dg11I = self.xp.ravel(self.xp.kron(dg11, self.xp.array([0, 1])))

            # put into sparse matrix
            dg = sp.diags(dg00I, offsets=0) + sp.diags(dg11I, offsets=0)
            dg += sp.diags(dg01I, offsets=1, shape=dg.shape) + sp.diags(dg10I, offsets=-1, shape=dg.shape)

            # interleave g terms to apply inverse to it
            g = self.xp.kron(tmpgu.flatten(), self.xp.array([1, 0])) + self.xp.kron(
                tmpgv.flatten(), self.xp.array([0, 1])
            )
            # invert dg matrix
            b = sp.linalg.spsolve(dg, g)
            # update real space vectors
            tmpu[:] -= b[::2].reshape(self.nvars)
            tmpv[:] -= b[1::2].reshape(self.nvars)

            # increase iteration count
            n += 1
            self.work_counters['newton']()

        if self.xp.isnan(res) and self.stop_at_nan:
            raise ProblemError('Newton got nan after %i iterations, aborting...' % n)
        elif self.xp.isnan(res):
            self.logger.warning('Newton got nan after %i iterations...' % n)

        if n == self.newton_maxiter:
            self.logger.warning('Newton did not converge after %i iterations, error is %s' % (n, res))

        me = self.dtype_u(self.init)
        if self.spectral:
            me[0, ...] = self.fft.forward(tmpu)
            me[1, ...] = self.fft.forward(tmpv)
        else:
            me[0, ...] = tmpu
            me[1, ...] = tmpv
        return me


class grayscott_mi_linear(grayscott_imex_linear):
    r"""
    The original Gray-Scott system [1]_ describes a reaction-diffusion process of two substances :math:`u` and :math:`v`,
    where they diffuse over time. During the reaction :math:`u` is used up with overall decay rate :math:`B`,
    whereas :math:`v` is produced with feed rate :math:`A`. :math:`D_u,\, D_v` are the diffusion rates for
    :math:`u,\, v`. The model with linear (reaction) part is described by the :math:`N`-dimensional model

    .. math::
        \frac{\partial u}{\partial t} = D_u \Delta u - u v^2 + A,

    .. math::
        \frac{\partial v}{\partial t} = D_v \Delta v + u v^2

    in :math:`x \in \Omega:=[-L/2, L/2]^N` with :math:`N=2,3`. Spatial discretization is done by using
    Fast Fourier transformation for solving the linear parts provided by ``mpi4py-fft`` [2]_, see also
    https://mpi4py-fft.readthedocs.io/en/latest/.

    The problem in this class will be treated in a *multi-implicit* way for time-stepping, i.e., for the system containing
    the diffusion part will be solved by FFT, and for the linear part a Newton solver is used.
    """

    dtype_f = comp2_mesh

    def __init__(
        self,
        newton_maxiter=100,
        newton_tol=1e-12,
        **kwargs,
    ):
        """Initialization routine"""
        super().__init__(**kwargs)
        # This may not run in parallel yet..
        assert self.comm.Get_size() == 1
        self.work_counters['newton'] = WorkCounter()
        self.Ku = -self.Du * self.K2 - self.A
        self.Kv = -self.Dv * self.K2 - self.B
        self._makeAttributeAndRegister('newton_maxiter', 'newton_tol', localVars=locals(), readOnly=False)

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
            f.comp1[0, ...] = self.Ku * u[0, ...]
            f.comp1[1, ...] = self.Kv * u[1, ...]
            tmpu = newDistArray(self.fft, False)
            tmpv = newDistArray(self.fft, False)
            tmpu[:] = self.fft.backward(u[0, ...], tmpu)
            tmpv[:] = self.fft.backward(u[1, ...], tmpv)
            tmpfu = -tmpu * tmpv**2 + self.A
            tmpfv = tmpu * tmpv**2
            f.comp2[0, ...] = self.fft.forward(tmpfu)
            f.comp2[1, ...] = self.fft.forward(tmpfv)

        else:
            u_hat = self.fft.forward(u[0, ...])
            lap_u_hat = self.Ku * u_hat
            f.comp1[0, ...] = self.fft.backward(lap_u_hat, f.comp1[0, ...])
            u_hat = self.fft.forward(u[1, ...])
            lap_u_hat = self.Kv * u_hat
            f.comp1[1, ...] = self.fft.backward(lap_u_hat, f.comp1[1, ...])
            f.comp2[0, ...] = -u[0, ...] * u[1, ...] ** 2 + self.A
            f.comp2[1, ...] = u[0, ...] * u[1, ...] ** 2

        self.work_counters['rhs']()
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
            tmpu[:] = self.fft.backward(u[0, ...], tmpu)
            tmpv[:] = self.fft.backward(u[1, ...], tmpv)
            tmprhsu = newDistArray(self.fft, False)
            tmprhsv = newDistArray(self.fft, False)
            tmprhsu[:] = self.fft.backward(rhs[0, ...], tmprhsu)
            tmprhsv[:] = self.fft.backward(rhs[1, ...], tmprhsv)

        else:
            tmpu = u[0, ...]
            tmpv = u[1, ...]
            tmprhsu = rhs[0, ...]
            tmprhsv = rhs[1, ...]

        # start newton iteration
        n = 0
        res = 99
        while n < self.newton_maxiter:
            # print(n, res)
            # form the function g with g(u) = 0
            tmpgu = tmpu - tmprhsu - factor * (-tmpu * tmpv**2 + self.A)
            tmpgv = tmpv - tmprhsv - factor * (tmpu * tmpv**2)

            # if g is close to 0, then we are done
            res = max(self.xp.linalg.norm(tmpgu, self.xp.inf), self.xp.linalg.norm(tmpgv, self.xp.inf))
            if res < self.newton_tol:
                break

            # assemble dg
            dg00 = 1 - factor * (-(tmpv**2))
            dg01 = -factor * (-2 * tmpu * tmpv)
            dg10 = -factor * (tmpv**2)
            dg11 = 1 - factor * (2 * tmpu * tmpv)

            # interleave and unravel to put into sparse matrix
            dg00I = self.xp.ravel(self.xp.kron(dg00, self.xp.array([1, 0])))
            dg01I = self.xp.ravel(self.xp.kron(dg01, self.xp.array([1, 0])))
            dg10I = self.xp.ravel(self.xp.kron(dg10, self.xp.array([1, 0])))
            dg11I = self.xp.ravel(self.xp.kron(dg11, self.xp.array([0, 1])))

            # put into sparse matrix
            dg = sp.diags(dg00I, offsets=0) + sp.diags(dg11I, offsets=0)
            dg += sp.diags(dg01I, offsets=1, shape=dg.shape) + sp.diags(dg10I, offsets=-1, shape=dg.shape)

            # interleave g terms to apply inverse to it
            g = self.xp.kron(tmpgu.flatten(), self.xp.array([1, 0])) + self.xp.kron(
                tmpgv.flatten(), self.xp.array([0, 1])
            )
            # invert dg matrix
            b = sp.linalg.spsolve(dg, g)
            # update real-space vectors
            tmpu[:] -= b[::2].reshape(self.nvars)
            tmpv[:] -= b[1::2].reshape(self.nvars)

            # increase iteration count
            n += 1
            self.work_counters['newton']()

        if self.xp.isnan(res) and self.stop_at_nan:
            raise ProblemError('Newton got nan after %i iterations, aborting...' % n)
        elif self.xp.isnan(res):
            self.logger.warning('Newton got nan after %i iterations...' % n)

        if n == self.newton_maxiter:
            self.logger.warning('Newton did not converge after %i iterations, error is %s' % (n, res))

        me = self.dtype_u(self.init)
        if self.spectral:
            me[0, ...] = self.fft.forward(tmpu)
            me[1, ...] = self.fft.forward(tmpv)
        else:
            me[0, ...] = tmpu
            me[1, ...] = tmpv
        return me
