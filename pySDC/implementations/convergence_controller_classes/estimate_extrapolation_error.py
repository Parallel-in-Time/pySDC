import numpy as np
from scipy.special import factorial

from pySDC.core.ConvergenceController import ConvergenceController
from pySDC.core.Errors import DataError
from pySDC.implementations.datatype_classes.mesh import mesh, imex_mesh


class EstimateExtrapolationError(ConvergenceController):

    def setup(self, controller, params, description):
        """
        The extrapolation based method requires storage of previous values of u, f, t and dt and also requires solving
        a linear system of equations to compute the Taylor expansion finite difference style. Here, all variables are
        initialized which are needed for this process.
        """

        from pySDC.implementations.convergence_controller_classes.hotrod import HotRod
        from pySDC.implementations.convergence_controller_classes.adaptivity import Adaptivity

        default_params = {
            'control_order': -75,
            'use_adaptivity': any([me == Adaptivity for me in description.get('convergence_controllers', {})]),
            'use_HotRod': any([me == HotRod for me in description.get('convergence_controllers', {})]),
            'order_time_marching': description['step_params']['maxiter'],
        }

        # Do a sufficiently high order Taylor expansion
        default_params['Taylor_order'] = default_params['order_time_marching'] + 2

        # Estimate and store values from this iteration
        default_params['estimate_iter'] = default_params['order_time_marching'] - (1 if default_params['use_HotRod']
                                                                                   else 0)

        # Store n values. Since we store u and f, we need only half of each (the +1 is for rounding)
        default_params['n'] = (default_params['Taylor_order'] + 1) // 2
        default_params['n_per_proc'] = default_params['n'] * 1

        self.u_coeff = [None] * default_params['n']
        self.f_coeff = [0.] * default_params['n']

        return default_params | params

    def check_parameters(self, controller, params, description):
        '''
        Check whether parameters are compatible with whatever assumptions went into the step size functions etc.

        Returns:
            Whether the parameters are compatible and an error text if not
        '''
        if description['step_params'].get('restol', -1.) >= 0:
            return False, 'Extrapolation error needs constant order in time and hence restol in the step parameters \
has to be smaller than 0!'

        if controller.params.mssdc_jac:
            return False, 'Extrapolation error estimator needs the same order on all steps, please activate Gauss-Seid\
el multistep mode!'

        return True, ''

    def store_values(self, S):
        """
        Store the required attributes of the step to do the extrapolation. We only care about the last collocation
        node on the finest level at the moment.
        """
        # figure out which values are to be replaced by the new ones
        if None in self.t:
            oldest_val = len(self.t) - len(self.t[self.t == [None]])
        else:
            oldest_val = np.argmin(self.t)

        f = S.levels[0].f[-1]
        if type(f) == imex_mesh:
            self.f[oldest_val] = f.impl + f.expl
        elif type(f) == mesh:
            self.f[oldest_val] = f
        else:
            raise DataError(f'Unable to store f from datatype {type(f)}, extrapolation based error estimate only\
            works with types imex_mesh and mesh')

        self.u[oldest_val] = S.levels[0].u[-1]
        self.t[oldest_val] = S.time + S.dt
        self.dt[oldest_val] = S.dt

    def get_extrapolation_coefficients(self, t, dt, t_eval):
        """
        This function solves a linear system where in the matrix A, the row index reflects the order of the derivative
        in the Taylor expansion and the column index reflects the particular step and whether its u or f from that
        step. The vector b on the other hand, contains a 1 in the first entry and zeros elsewhere, since we want to
        compute the value itself and all the derivatives should vanish after combining the Taylor expansions. This
        works to the order the number of rows and since we want a square matrix for solving, we need the same amount of
        colums, which determines the memory overhead, since it is equal to the solutions / rhs that we need in memory
        at the time of evaluation.
        This is enough to get the extrapolated solution, but if we want to compute the local error, we have to compute
        a prefactor. This is based on error accumulation between steps (first step's solution is exact plus 1 LTE,
        second solution is exact plus 2 LTE and so on), which can be computed for adaptive step sizes as well, but its
        wonky in time-parallel versions to say the least (it's not cared for and hence wrong, but it might be wrong in
        the same way as the embedded method and work for Hot Rod regardless...)
        """

        # prepare A matrix
        A = np.zeros((self.params.Taylor_order, self.params.Taylor_order))
        A[0, 0:self.params.n] = 1.
        j = np.arange(self.params.Taylor_order)
        inv_facs = 1. / factorial(j)

        # get the steps backwards from the point of evaluation
        idx = np.argsort(t)
        steps_from_now = t[idx] - t_eval

        # fill A matrix
        for i in range(1, self.params.Taylor_order):
            # Taylor expansions of the solutions
            A[i, :self.params.n] = steps_from_now**j[i] * inv_facs[i]

            # Taylor expansions of the first derivatives a.k.a. right hand side evaluations
            A[i, self.params.n:self.params.Taylor_order] =\
                steps_from_now[2 * self.params.n - self.params.Taylor_order:]**(j[i] - 1) * inv_facs[i - 1]

        # prepare rhs
        b = np.zeros(self.params.Taylor_order)
        b[0] = 1.

        # solve linear system for the coefficients
        coeff = np.linalg.solve(A, b)
        self.u_coeff = coeff[:self.params.n]
        self.f_coeff[self.params.n * 2 - self.params.Taylor_order:] = coeff[self.params.n:self.params.Taylor_order]

        # determine prefactor
        r = abs(dt[len(dt) - len(self.u_coeff):] / dt[-1])**(self.params.Taylor_order - 1)
        inv_prefactor = -sum(r[1:]) - 1.
        for i in range(len(self.u_coeff)):
            inv_prefactor += sum(r[1: i + 1]) * self.u_coeff[i]
        self.prefactor = 1. / abs(inv_prefactor)


class EstimateExtrapolationErrorNonMPI(EstimateExtrapolationError):
    def setup(self, controller, params, description):
        '''
        Add a no parameter 'no_storage' which decides whether the standart or the no-memory-overhead version is run,
        where only values are used for extrapolation which are in memory of other processes
        '''
        default_params = super(EstimateExtrapolationErrorNonMPI, self).setup(controller, params, description)

        non_mpi_defaults = {
            'no_storage': False,
        }

        self.t = np.array([None] * default_params['n'])
        self.dt = np.array([None] * default_params['n'])
        self.u = [None] * default_params['n']
        self.f = [None] * default_params['n']

        return non_mpi_defaults | default_params

    def post_iteration_processing(self, controller, S):
        if S.status.iter == self.params.estimate_iter:
            t_eval = S.time + S.dt

            # compute the extrapolation coefficients if needed
            if (None in self.u_coeff or self.params.use_adaptivity) and\
                    None not in self.t and t_eval > max(self.t):
                self.get_extrapolation_coefficients(self.t, self.dt, t_eval)

            # compute the error if we can
            if None not in self.u_coeff and None not in self.t:
                self.get_extrapolated_error(S)

            # store the solution and pretend we didn't because in the non MPI version we take a few shortcuts
            if self.params.no_storage:
                self.store_values(S)

    def prepare_next_block_nonMPI(self, controller, MS, active_slots, time, Tend):
        '''
        If the no-memory-overhead version is used, we need to delete stuff that shouldn't be available otherwise, we
        need to store all stuff we can.
        '''
        if self.params.no_storage:
            self.t = np.array([None] * self.params.n)
            self.dt = np.array([None] * self.params.n)
            self.u = [None] * self.params.n
            self.f = [None] * self.params.n

        else:
            # decide where we need to restart to store everything up to that point
            MS_active = [MS[i] for i in range(len(MS)) if i in active_slots]
            restarts = [S.status.restart for S in MS_active]
            restart_at = np.where(restarts)[0][0] if True in restarts else len(MS_active)

            # store values in the current block that don't need restarting
            if restart_at > 0:
                [self.store_values(S) for S in MS_active[:restart_at]]

    def get_extrapolated_solution(self, S):
        '''
        Combine values from previous steps to extrapolate.
        '''
        if len(S.levels) > 1:
            raise NotImplementedError('Extrapolated estimate only works on the finest level for now')

        # prepare variables
        u_ex = S.levels[0].u[-1] * 0.
        idx = np.argsort(self.t)

        # see if we have a solution for the current step already stored
        if (abs(S.time + S.dt - self.t) < 10. * np.finfo(float).eps).any():
            idx_step = idx[np.argmin(abs(self.t - S.time - S.dt))]
        else:
            idx_step = max(idx) + 1

        # make a mask of all the steps we want to include in the extrapolation
        mask = np.logical_and(idx < idx_step, idx >= idx_step - self.params.n)

        # do the extrapolation by summing everything up
        for i in range(self.params.n):
            u_ex += self.u_coeff[i] * self.u[idx[mask][i]] + self.f_coeff[i] * self.f[idx[mask][i]]

        return u_ex

    def get_extrapolated_error(self, S):
        """
        The extrapolation estimate combines values of u and f from multiple steps to extrapolate and compare to the
        solution obtained by the time marching scheme.
        """
        u_ex = self.get_extrapolated_solution(S)
        if u_ex is not None:
            S.levels[0].status.error_extrapolation_estimate = abs(u_ex - S.levels[0].u[-1]) * self.prefactor
        else:
            S.levels[0].status.error_extrapolation_estimate = None
