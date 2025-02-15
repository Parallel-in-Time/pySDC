import numpy as np
import scipy.sparse as nsp

from pySDC.core.problem import Problem, WorkCounter
from pySDC.implementations.datatype_classes.mesh import mesh, imex_mesh


class testequation0d(Problem):
    r"""
    This class implements the simple test equation of the form

    .. math::
        \frac{d u(t)}{dt} = A u(t)

    for :math:`A = diag(\lambda_1, .. ,\lambda_n)`.

    Parameters
    ----------
    lambdas : sequence of array_like, optional
        List of lambda parameters.
    u0 : sequence of array_like, optional
        Initial condition.

    Attributes
    ----------
    A : scipy.sparse.csc_matrix
        Diagonal matrix containing :math:`\lambda_1,..,\lambda_n`.
    """

    xp = np
    xsp = nsp
    dtype_u = mesh
    dtype_f = mesh

    @classmethod
    def setup_GPU(cls):
        """
        Switch to GPU modules
        """
        from pySDC.implementations.datatype_classes.cupy_mesh import cupy_mesh
        import cupy as cp
        import cupyx.scipy.sparse as csp

        cls.xp = cp
        cls.xsp = csp
        cls.dtype_u = cupy_mesh
        cls.dtype_f = cupy_mesh

    def __init__(self, lambdas=None, u0=0.0, useGPU=False):
        """Initialization routine"""
        if useGPU:
            self.setup_GPU()

        if lambdas is None:
            re = self.xp.linspace(-30, 19, 50)
            im = self.xp.linspace(-50, 49, 50)
            lambdas = self.xp.array([[complex(re[i], im[j]) for i in range(len(re))] for j in range(len(im))]).reshape(
                (len(re) * len(im))
            )
        lambdas = self.xp.asarray(lambdas)
        assert lambdas.ndim == 1, f'expect flat list here, got {lambdas}'
        nvars = lambdas.size
        assert nvars > 0, 'expect at least one lambda parameter here'

        # invoke super init, passing number of dofs, dtype_u and dtype_f
        super().__init__(init=(nvars, None, self.xp.dtype('complex128')))

        lambdas = self.xp.array(lambdas)
        self.A = self.xsp.diags(lambdas)
        self._makeAttributeAndRegister('nvars', 'lambdas', 'u0', 'useGPU', localVars=locals(), readOnly=True)
        self.work_counters['rhs'] = WorkCounter()

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
        f[:] = u
        f *= self.lambdas
        self.work_counters['rhs']()
        return f

    def solve_system(self, rhs, factor, u0, t):
        r"""
        Simple linear solver for :math:`(I-factor\cdot A)\vec{u}=\vec{rhs}`.

        Parameters
        ----------
        rhs : dtype_f
            Right-hand side for the linear system.
        factor : float
            Abbrev. for the local stepsize (or any other factor required).
        u0 : dtype_u
            Initial guess for the iterative solver.
        t : float
            Current time (e.g. for time-dependent BCs).

        Returns
        -------
        me : dtype_u
            The solution as mesh.
        """
        me = self.dtype_u(self.init)
        L = 1 - factor * self.lambdas
        L[L == 0] = 1  # to avoid potential divisions by zeros
        me[:] = rhs
        me /= L
        return me

    def u_exact(self, t, u_init=None, t_init=None):
        """
        Routine to compute the exact solution at time t.

        Parameters
        ----------
        t : float
            Time of the exact solution.
        u_init : pySDC.problem.testequation0d.dtype_u
            Initial solution.
        t_init : float
            The initial time.

        Returns
        -------
        me : dtype_u
            The exact solution.
        """

        u_init = (self.u0 if u_init is None else u_init) * 1.0
        t_init = 0.0 if t_init is None else t_init * 1.0

        me = self.dtype_u(self.init)
        me[:] = u_init * self.xp.exp((t - t_init) * self.lambdas)
        return me


class test_equation_IMEX(Problem):
    dtype_f = imex_mesh
    dtype_u = mesh
    xp = np
    xsp = nsp

    def __init__(self, lambdas_implicit=None, lambdas_explicit=None, u0=0.0):
        """Initialization routine"""

        if lambdas_implicit is None:
            re = self.xp.linspace(-30, 19, 50)
            im = self.xp.linspace(-50, 49, 50)
            lambdas_implicit = self.xp.array(
                [[complex(re[i], im[j]) for i in range(len(re))] for j in range(len(im))]
            ).reshape((len(re) * len(im)))
        if lambdas_explicit is None:
            re = self.xp.linspace(-30, 19, 50)
            im = self.xp.linspace(-50, 49, 50)
            lambdas_implicit = self.xp.array(
                [[complex(re[i], im[j]) for i in range(len(re))] for j in range(len(im))]
            ).reshape((len(re) * len(im)))
        lambdas_implicit = self.xp.asarray(lambdas_implicit)
        lambdas_explicit = self.xp.asarray(lambdas_explicit)

        assert lambdas_implicit.ndim == 1, f'expect flat list here, got {lambdas_implicit}'
        assert lambdas_explicit.shape == lambdas_implicit.shape
        nvars = lambdas_implicit.size
        assert nvars > 0, 'expect at least one lambda parameter here'

        # invoke super init, passing number of dofs, dtype_u and dtype_f
        super().__init__(init=(nvars, None, self.xp.dtype('complex128')))

        self.A = self.xsp.diags(lambdas_implicit)
        self._makeAttributeAndRegister(
            'nvars', 'lambdas_implicit', 'lambdas_explicit', 'u0', localVars=locals(), readOnly=True
        )
        self.work_counters['rhs'] = WorkCounter()

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
        f.impl[:] = u * self.lambdas_implicit
        f.expl[:] = u * self.lambdas_explicit
        self.work_counters['rhs']()
        return f

    def solve_system(self, rhs, factor, u0, t):
        r"""
        Simple linear solver for :math:`(I-factor\cdot A)\vec{u}=\vec{rhs}`.

        Parameters
        ----------
        rhs : dtype_f
            Right-hand side for the linear system.
        factor : float
            Abbrev. for the local stepsize (or any other factor required).
        u0 : dtype_u
            Initial guess for the iterative solver.
        t : float
            Current time (e.g. for time-dependent BCs).

        Returns
        -------
        me : dtype_u
            The solution as mesh.
        """
        me = self.dtype_u(self.init)
        L = 1 - factor * self.lambdas_implicit
        L[L == 0] = 1  # to avoid potential divisions by zeros
        me[:] = rhs
        me /= L
        return me

    def u_exact(self, t, u_init=None, t_init=None):
        """
        Routine to compute the exact solution at time t.

        Parameters
        ----------
        t : float
            Time of the exact solution.
        u_init : pySDC.problem.testequation0d.dtype_u
            Initial solution.
        t_init : float
            The initial time.

        Returns
        -------
        me : dtype_u
            The exact solution.
        """

        u_init = (self.u0 if u_init is None else u_init) * 1.0
        t_init = 0.0 if t_init is None else t_init * 1.0

        me = self.dtype_u(self.init)
        me[:] = u_init * self.xp.exp((t - t_init) * (self.lambdas_implicit + self.lambdas_explicit))
        return me
