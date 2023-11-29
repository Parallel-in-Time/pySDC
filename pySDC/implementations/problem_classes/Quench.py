import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve, gmres
from scipy.linalg import inv

from pySDC.core.Errors import ProblemError
from pySDC.core.Problem import ptype, WorkCounter
from pySDC.helpers import problem_helper
from pySDC.implementations.datatype_classes.mesh import mesh, imex_mesh


# noinspection PyUnusedLocal
class Quench(ptype):
    """
    This is a toy problem [1]_ to emulate a magnet that has been cooled to temperatures where superconductivity is possible.
    However, there is a leak! Some point in the domain is constantly heated and when this has heated up its environment
    sufficiently, there will be a runaway effect heating up the entire magnet.
    This effect has actually lead to huge magnets being destroyed at CERN in the past and hence warrants investigation.

    The model we use is a 1d heat equation with Neumann-zero boundary conditions, meaning this magnet is totally
    insulated from its environment except for the leak.
    We add a non-linear term that heats parts of the domain that exceed a certain temperature threshold as well as the
    leak itself.

    The problem is discretised with finite difference in space and treated *fully-implicitly*.

    Parameters
    ----------
    Cv : float, optional
        Volumetric heat capacity.
    K : float, optional
        Thermal conductivity.
    u_thresh : float, optional
        Threshold for temperature.
    u_max : float, optional
        Maximum temperature.
    Q_max : float, optional
        Maximum heat source power density.
    leak_range : tuple of float
        Range of the leak.
    leak_type : str, optional
        Type of leak, choose between ``'linear'`` or ``'exponential'``.
    leak_transition : str, optional
        Indicates how the heat in the leak propagates, choose between ``'step'`` and ``'Gaussian'``.
    order : int, optional
        Order of the finite difference discretization.
    stencil_type : str, optional
        Type of stencil for finite differences.
    bc : str, optional
        Type of boundary conditions. Default is ``'neumann-zero'``.
    nvars : int, optional
        Spatial resolution.
    newton_tol : float, optional
        Tolerance for Newton to terminate.
    newton_maxiter : int, optional
        Maximum number of Newton iterations to be done.
    lintol : float, optional
        Tolerance for linear solver to be done.
    liniter : int, optional
        Maximum number of linear iterations inside the Newton solver.
    direct_solver : bool, optional
        Indicates if a direct solver should be used.
    inexact_linear_ratio : float, optional
        Ratio of tolerance of linear solver to the Newton residual, overrides `lintol`
    min_lintol : float, optional
        Minimal tolerance for the linear solver
    reference_sol_type : str, optional
        Indicates which method should be used to compute a reference solution.
        Choose between ``'scipy'``, ``'SDC'``, or ``'DIRK'``.

    Attributes
    ----------
    A : sparse matrix (CSC)
        FD discretization matrix of the ND grad operator.
    Id : sparse matrix (CSC)
        Identity matrix of the same dimension as A.
    dx : float
        Distance between two spatial nodes.
    xv : np.1darray
        Spatial grid values.
    leak : np.1darray of bool
        Indicates the leak.

    References
    ----------
    .. [1] Thermal thin shell approximation towards finite element quench simulation. E. Schnaubelt, M. Wozniak, S. Schöps.
       Supercond. Sci. Technol. 36 044004. DOI 10.1088/1361-6668/acbeea
    """

    dtype_u = mesh
    dtype_f = mesh

    def __init__(
        self,
        Cv=1000.0,
        K=1000.0,
        u_thresh=3e-2,
        u_max=6e-2,
        Q_max=1.0,
        leak_range=(0.45, 0.55),
        leak_type='linear',
        leak_transition='step',
        order=2,
        stencil_type='center',
        bc='neumann-zero',
        nvars=2**7,
        newton_tol=1e-8,
        newton_maxiter=99,
        lintol=1e-8,
        liniter=99,
        direct_solver=True,
        inexact_linear_ratio=None,
        min_lintol=1e-12,
        reference_sol_type='scipy',
    ):
        """
        Initialization routine

        Args:
            problem_params (dict): custom parameters for the example
            dtype_u: mesh data type (will be passed parent class)
            dtype_f: mesh data type (will be passed parent class)
        """
        # invoke super init, passing number of dofs, dtype_u and dtype_f
        super().__init__(init=(nvars, None, np.dtype('float64')))
        self._makeAttributeAndRegister(
            'Cv',
            'K',
            'u_thresh',
            'u_max',
            'Q_max',
            'leak_range',
            'leak_type',
            'leak_transition',
            'order',
            'stencil_type',
            'bc',
            'nvars',
            'direct_solver',
            'reference_sol_type',
            localVars=locals(),
            readOnly=True,
        )
        self._makeAttributeAndRegister(
            'newton_tol',
            'newton_maxiter',
            'lintol',
            'liniter',
            'inexact_linear_ratio',
            'min_lintol',
            localVars=locals(),
            readOnly=False,
        )

        self._makeAttributeAndRegister(
            'newton_tol',
            'newton_maxiter',
            'lintol',
            'liniter',
            'direct_solver',
            localVars=locals(),
            readOnly=False,
        )

        # setup finite difference discretization from problem helper
        self.dx, xvalues = problem_helper.get_1d_grid(size=self.nvars, bc=self.bc)

        self.A, self.b = problem_helper.get_finite_difference_matrix(
            derivative=2,
            order=self.order,
            stencil_type=self.stencil_type,
            dx=self.dx,
            size=self.nvars,
            dim=1,
            bc=self.bc,
        )
        self.A *= self.K / self.Cv

        self.xv = xvalues
        self.Id = sp.eye(np.prod(self.nvars), format='csc')

        self.leak = np.logical_and(self.xv > self.leak_range[0], self.xv < self.leak_range[1])

        self.work_counters['newton'] = WorkCounter()
        self.work_counters['rhs'] = WorkCounter()
        if not self.direct_solver:
            self.work_counters['linear'] = WorkCounter()

    def eval_f_non_linear(self, u, t):
        """
        Get the non-linear part of f.

        Parameters
        ----------
        u : dtype_u
            Current values of the numerical solution:
        t : float
            Current time at which the numerical solution is computed.

        Returns
        -------
        me : dtype_u
            The non-linear part of the right-hand side.
        """
        u_thresh = self.u_thresh
        u_max = self.u_max
        Q_max = self.Q_max
        me = self.dtype_u(self.init)

        if self.leak_type == 'linear':
            me[:] = (u - u_thresh) / (u_max - u_thresh) * Q_max
        elif self.leak_type == 'exponential':
            me[:] = Q_max * (np.exp(u) - np.exp(u_thresh)) / (np.exp(u_max) - np.exp(u_thresh))
        else:
            raise NotImplementedError(f'Leak type \"{self.leak_type}\" not implemented!')

        me[u < u_thresh] = 0
        if self.leak_transition == 'step':
            me[self.leak] = Q_max
        elif self.leak_transition == 'Gaussian':
            me[:] = np.max([me, Q_max * np.exp(-((self.xv - 0.5) ** 2) / 3e-2)], axis=0)
        else:
            raise NotImplementedError(f'Leak transition \"{self.leak_transition}\" not implemented!')

        me[u >= u_max] = Q_max

        me[:] /= self.Cv

        return me

    def eval_f(self, u, t):
        """
        Evaluate the full right-hand side.

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
        f[:] = self.A.dot(u.flatten()).reshape(self.nvars) + self.b + self.eval_f_non_linear(u, t)

        self.work_counters['rhs']()
        return f

    def get_non_linear_Jacobian(self, u):
        """
        Evaluate the non-linear part of the Jacobian only.

        Parameters
        ----------
        u : dtype_u
            Current values of the numerical solution.

        Returns
        -------
        scipy.sparse.csc
            The derivative of the non-linear part of the solution w.r.t. to the solution.
        """
        u_thresh = self.u_thresh
        u_max = self.u_max
        Q_max = self.Q_max
        me = self.dtype_u(self.init)

        if self.leak_type == 'linear':
            me[:] = Q_max / (u_max - u_thresh)
        elif self.leak_type == 'exponential':
            me[:] = Q_max * np.exp(u) / (np.exp(u_max) - np.exp(u_thresh))
        else:
            raise NotImplementedError(f'Leak type {self.leak_type} not implemented!')

        me[u < u_thresh] = 0
        if self.leak_transition == 'step':
            me[self.leak] = 0
        elif self.leak_transition == 'Gaussian':
            me[self.leak] = 0
            me[self.leak][u[self.leak] > Q_max * np.exp(-((self.xv[self.leak] - 0.5) ** 2) / 3e-2)] = 1
        else:
            raise NotImplementedError(f'Leak transition \"{self.leak_transition}\" not implemented!')
        me[u > u_max] = 0

        me[:] /= self.Cv

        return sp.diags(me, format='csc')

    def solve_system(self, rhs, factor, u0, t):
        r"""
        Simple Newton solver for :math:`(I - factor \cdot f)(\vec{u}) = \vec{rhs}`.

        Parameters
        ----------
        rhs : dtype_f
            Right-hand side.
        factor : float
            Abbrev. for the local stepsize (or any other factor required).
        u0 : dtype_u
            Initial guess for the iterative solver.
        t : float
            Current time (e.g. for time-dependent BCs).

        Returns
        -------
        u : dtype_u
            The solution as mesh.
        """
        u = self.dtype_u(u0)
        res = np.inf
        delta = self.dtype_u(self.init, val=0.0)

        # construct a preconditioner for the space solver
        if not self.direct_solver:
            M = inv((self.Id - factor * self.A).toarray())
            zero = self.dtype_u(self.init, val=0.0)

        for n in range(0, self.newton_maxiter):
            # assemble G such that G(u) = 0 at the solution of the step
            G = u - factor * self.eval_f(u, t) - rhs
            self.work_counters[
                'rhs'
            ].decrement()  # Work regarding construction of the Jacobian etc. should count into the Newton iterations only

            res = np.linalg.norm(G, np.inf)
            if res <= self.newton_tol and n > 0:  # we want to make at least one Newton iteration
                break

            if self.inexact_linear_ratio:
                self.lintol = max([res * self.inexact_linear_ratio, self.min_lintol])

            # assemble Jacobian J of G
            J = self.Id - factor * (self.A + self.get_non_linear_Jacobian(u))

            # solve the linear system
            if self.direct_solver:
                delta = spsolve(J, G)
            else:
                delta, info = gmres(
                    J,
                    G,
                    x0=zero,
                    M=M,
                    tol=self.lintol,
                    maxiter=self.liniter,
                    atol=0,
                    callback=self.work_counters['linear'],
                )

            if not np.isfinite(delta).all():
                break

            # update solution
            u = u - delta

            self.work_counters['newton']()

        return u

    def u_exact(self, t, u_init=None, t_init=None):
        r"""
        Routine to compute the exact solution at time :math:`t`.

        Parameters
        ----------
        t : float
            Time of the exact solution.

        Returns
        -------
        me : dtype_u
            The exact solution.
        """

        me = self.dtype_u(self.init, val=0.0)

        if t > 0:
            if self.reference_sol_type == 'scipy':

                def jac(t, u):
                    """
                    Get the Jacobian for the implicit BDF method to use in `scipy.solve_ivp`

                    Parameters
                    ----------
                    t : float
                        The current time.
                    u : dtype_u
                        Current solution.

                    Returns
                    -------
                    scipy.sparse.csc
                        The derivative of the non-linear part of the solution w.r.t. to the solution.
                    """
                    return self.A + self.get_non_linear_Jacobian(u)

                def eval_rhs(t, u):
                    """
                    Function to pass to `scipy.solve_ivp` to evaluate the full right-hand side.

                    Parameters
                    ----------
                    t : float
                        Current time.
                    u : numpy.1darray
                        Current solution.

                    Returns
                    -------
                    numpy.1darray
                        Right-hand side.
                    """
                    return self.eval_f(u.reshape(self.init[0]), t).flatten()

                me[:] = self.generate_scipy_reference_solution(eval_rhs, t, u_init, t_init, method='BDF', jac=jac)

            elif self.reference_sol_type in ['DIRK', 'SDC']:
                from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
                from pySDC.implementations.hooks.log_solution import LogSolution
                from pySDC.helpers.stats_helper import get_sorted

                description = {}
                description['problem_class'] = Quench
                description['problem_params'] = {
                    'newton_tol': 1e-10,
                    'newton_maxiter': 99,
                    'nvars': 2**10,
                    **self.params,
                }

                if self.reference_sol_type == 'DIRK':
                    from pySDC.implementations.sweeper_classes.Runge_Kutta import DIRK43
                    from pySDC.implementations.convergence_controller_classes.adaptivity import AdaptivityRK

                    description['sweeper_class'] = DIRK43
                    description['sweeper_params'] = {}
                    description['step_params'] = {'maxiter': 1}
                    description['level_params'] = {'dt': 1e-4}
                    description['convergence_controllers'] = {AdaptivityRK: {'e_tol': 1e-9, 'update_order': 4}}
                elif self.reference_sol_type == 'SDC':
                    from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit

                    description['sweeper_class'] = generic_implicit
                    description['sweeper_params'] = {'num_nodes': 3, 'QI': 'IE', 'quad_type': 'RADAU-RIGHT'}
                    description['step_params'] = {'maxiter': 99}
                    description['level_params'] = {'dt': 0.5, 'restol': 1e-10}

                controller_params = {'hook_class': LogSolution, 'mssdc_jac': False, 'logger_level': 99}

                controller = controller_nonMPI(
                    description=description, controller_params=controller_params, num_procs=1
                )

                uend, stats = controller.run(
                    u0=u_init if u_init is not None else self.u_exact(t=0.0),
                    t0=t_init if t_init is not None else 0,
                    Tend=t,
                )

                u_last = get_sorted(stats, type='u', recomputed=False)[-1]

                if abs(u_last[0] - t) > 1e-2:
                    self.logger.warning(
                        f'Time difference between reference solution and requested time is {abs(u_last[0]-t):.2e}!'
                    )

                me[:] = u_last[1]

        return me


class QuenchIMEX(Quench):
    """
    This is a toy problem [1]_ to emulate a magnet that has been cooled to temperatures where superconductivity is possible.
    However, there is a leak! Some point in the domain is constantly heated and when this has heated up its environment
    sufficiently, there will be a runaway effect heating up the entire magnet.
    This effect has actually lead to huge magnets being destroyed at CERN in the past and hence warrants investigation.

    The model we use is a 1d heat equation with Neumann-zero boundary conditions, meaning this magnet is totally
    insulated from its environment except for the leak.
    We add a non-linear term that heats parts of the domain that exceed a certain temperature threshold as well as the
    leak itself.

    The problem is discretised with finite difference in space and treated *semi-implicitly*.
    """

    dtype_f = imex_mesh

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
        f.impl[:] = self.A.dot(u.flatten()).reshape(self.nvars)
        f.expl[:] = self.eval_f_non_linear(u, t) + self.b

        self.work_counters['rhs']()
        return f

    def solve_system(self, rhs, factor, u0, t):
        r"""
        Simple linear solver for :math:`(I - factor \cdot f_{expl})(\vec{u}) = \vec{rhs}`.

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
        me[:] = spsolve(self.Id - factor * self.A, rhs.flatten()).reshape(self.nvars)
        return me

    def u_exact(self, t, u_init=None, t_init=None):
        r"""
        Routine to compute the exact solution at time :math:`t`.

        Parameters
        ----------
        t : float
            Time of the exact solution.

        Returns
        -------
        me : dtype_u
            The exact solution.
        """
        me = self.dtype_u(self.init, val=0.0)

        if t == 0:
            me[:] = super().u_exact(t, u_init, t_init)

        if t > 0:

            def jac(t, u):
                """
                Get the Jacobian for the implicit BDF method to use in `scipy.solve_ivp`.

                Parameters
                ----------
                t : float
                    Current time.
                u : dtype_u
                    Current solution.

                Returns
                -------
                scipy.sparse.csc
                    The derivative of the non-linear part of the solution w.r.t. to the solution.
                """
                return self.A

            def eval_rhs(t, u):
                """
                Function to pass to `scipy.solve_ivp` to evaluate the full right-hand side.

                Parameters
                ----------
                t : float
                    Current time
                u : numpy.1darray
                    Current solution

                Returns
                -------
                numpy.1darray
                    The right-hand side.
                """
                f = self.eval_f(u.reshape(self.init[0]), t)
                return (f.impl + f.expl).flatten()

            me[:] = self.generate_scipy_reference_solution(eval_rhs, t, u_init, t_init, method='BDF', jac=jac)
        return me
