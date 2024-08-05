import numpy as np
from scipy.optimize import newton_krylov
from scipy.optimize import NoConvergence

from pySDC.core.errors import ProblemError
from pySDC.core.problem import WorkCounter
from pySDC.implementations.problem_classes.generic_MPIFFT_Laplacian import IMEX_Laplacian_MPIFFT
from pySDC.implementations.datatype_classes.mesh import mesh


class nonlinearschroedinger_imex(IMEX_Laplacian_MPIFFT):
    r"""
    Example implementing the :math:`N`-dimensional nonlinear Schrödinger equation with periodic boundary conditions

    .. math::
        \frac{\partial u}{\partial t} = -i \Delta u + 2 c i |u|^2 u

    for fixed parameter :math:`c` and :math:`N=2, 3`. The linear parts of the problem will be solved using
    ``mpi4py-fft`` [1]_. *Semi-explicit* time-stepping is used here to solve the problem in the temporal dimension, i.e., the
    Laplacian will be handled implicitly.

    Parameters
    ----------
    nvars : tuple, optional
        Spatial resolution
    spectral : bool, optional
        If True, the solution is computed in spectral space.
    L : float, optional
        Denotes the period of the function to be approximated for the Fourier transform.
    c : float, optional
        Nonlinearity parameter.
    comm : MPI.COMM_World
        Communicator for parallelisation.

    Attributes
    ----------
    fft : PFFT
        Object for parallel FFT transforms.
    X : mesh-grid
        Grid coordinates in real space.
    K2 : matrix
        Laplace operator in spectral space.

    References
    ----------
    .. [1] Lisandro Dalcin, Mikael Mortensen, David E. Keyes. Fast parallel multidimensional FFT using advanced MPI.
        Journal of Parallel and Distributed Computing (2019).
    """

    def __init__(self, c=1.0, **kwargs):
        """Initialization routine"""
        super().__init__(L=2 * np.pi, alpha=1j, dtype='D', **kwargs)

        if not (c == 0.0 or c == 1.0):
            raise ProblemError(f'Setup not implemented, c has to be 0 or 1, got {c}')
        self._makeAttributeAndRegister('c', localVars=locals(), readOnly=True)

    def _eval_explicit_part(self, u, t, f_expl):
        f_expl[:] = self.ndim * self.c * 2j * self.xp.absolute(u) ** 2 * u
        return f_expl

    def u_exact(self, t, **kwargs):
        r"""
        Routine to compute the exact solution at time :math:`t`, see (1.3) https://arxiv.org/pdf/nlin/0702010.pdf for details

        Parameters
        ----------
        t : float
            Time of the exact solution.

        Returns
        -------
        u : dtype_u
            The exact solution.
        """
        if 'u_init' in kwargs.keys() or 't_init' in kwargs.keys():
            self.logger.warning(
                f'{type(self).__name__} uses an analytic exact solution from t=0. If you try to compute the local error, you will get the global error instead!'
            )

        def nls_exact_1D(t, x, c):
            ae = 1.0 / np.sqrt(2.0) * np.exp(1j * t)
            if c != 0:
                u = ae * ((np.cosh(t) + 1j * np.sinh(t)) / (np.cosh(t) - 1.0 / np.sqrt(2.0) * self.xp.cos(x)) - 1.0)
            else:
                u = self.xp.sin(x) * np.exp(-t * 1j)

            return u

        me = self.dtype_u(self.init, val=0.0)

        if self.spectral:
            tmp = nls_exact_1D(self.ndim * t, sum(self.X), self.c)
            me[:] = self.fft.forward(tmp)
        else:
            me[:] = nls_exact_1D(self.ndim * t, sum(self.X), self.c)

        return me


class nonlinearschroedinger_fully_implicit(nonlinearschroedinger_imex):
    r"""
    Example implementing the :math:`N`-dimensional nonlinear Schrödinger equation with periodic boundary conditions

    .. math::
        \frac{\partial u}{\partial t} = -i \Delta u + 2 c i |u|^2 u

    for fixed parameter :math:`c` and :math:`N=2, 3`. The linear parts of the problem will be discretized using
    ``mpi4py-fft`` [1]_. For time-stepping, the problem will be solved *fully-implicitly*, i.e., the nonlinear system containing
    the full right-hand side is solved by GMRES method.

    References
    ----------
    .. [1] https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.newton_krylov.html
    """

    dtype_u = mesh
    dtype_f = mesh

    def __init__(self, lintol=1e-9, liniter=99, **kwargs):
        super().__init__(**kwargs)
        self._makeAttributeAndRegister('liniter', 'lintol', localVars=locals(), readOnly=False)

        self.work_counters['newton'] = WorkCounter()

    def eval_f(self, u, t):
        """
        Routine to evaluate the right-hand side of the problem.

        Parameters
        ----------
        u : dtype_u
            Current values of the numerical solution.
        t : float
            Current time at which the numerical solution is computed.

        Returns
        -------
        f : dtype_f
            The right-hand side of the problem.
        """

        f = self.dtype_f(self.init)

        if self.spectral:
            tmp = self.fft.backward(u)
            tmpf = self.ndim * self.c * 2j * self.xp.absolute(tmp) ** 2 * tmp
            f[:] = -self.K2 * 1j * u + self.fft.forward(tmpf)

        else:
            u_hat = self.fft.forward(u)
            lap_u_hat = -self.K2 * 1j * u_hat
            f[:] = self.fft.backward(lap_u_hat) + self.ndim * self.c * 2j * self.xp.absolute(u) ** 2 * u

        self.work_counters['rhs']()
        return f

    def solve_system(self, rhs, factor, u0, t):
        r"""
        Solve the nonlinear system :math:`(1 - factor \cdot f)(\vec{u}) = \vec{rhs}` using a ``SciPy`` Newton-Krylov
        solver. See page [1]_ for details on the solver.

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
        me = self.dtype_u(self.init)

        # assemble the nonlinear function F for the solver
        def F(x):
            r"""
            Nonlinear function for the ``SciPy`` solver.

            Args:
                x : dtype_u
                    Current solution
            """
            self.work_counters['rhs'].decrement()
            return x - factor * self.eval_f(u=x.reshape(self.init[0]), t=t).reshape(x.shape) - rhs.reshape(x.shape)

        try:
            sol = newton_krylov(
                F=F,
                xin=u0.copy(),
                maxiter=self.liniter,
                x_tol=self.lintol,
                callback=self.work_counters['newton'],
                method='gmres',
            )
        except NoConvergence as e:
            sol = e.args[0]

        me[:] = sol
        return me
