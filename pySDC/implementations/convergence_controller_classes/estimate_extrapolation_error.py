import numpy as np
from scipy.special import factorial

from pySDC.core.ConvergenceController import ConvergenceController, Status
from pySDC.core.Errors import DataError
from pySDC.implementations.datatype_classes.mesh import mesh, imex_mesh


class EstimateExtrapolationErrorBase(ConvergenceController):
    '''
    Abstract base class for extrapolated error estimates
    ----------------------------------------------------
    This error estimate extrapolates a solution based on Taylor expansions using solutions of previous time steps.
    In particular, child classes need to implement how to make these solutions available, which works differently for
    MPI and non-MPI versions.
    '''

    def __init__(self, controller, params, description):
        '''
        Initialization routine

        Args:
            controller (pySDC.Controller): The controller
            params (dict): The params passed for this specific convergence controller
            description (dict): The description object used to instantiate the controller
        '''
        self.prev = Status(['t', 'u', 'f', 'dt'])  # store solutions etc. of previous steps here
        self.coeff = Status(['u', 'f', 'prefactor'])  # store coefficients for extrapolation here
        super(EstimateExtrapolationErrorBase, self).__init__(controller, params, description)

    def setup(self, controller, params, description):
        """
        The extrapolation based method requires storage of previous values of u, f, t and dt and also requires solving
        a linear system of equations to compute the Taylor expansion finite difference style. Here, all variables are
        initialized which are needed for this process.

        Args:
            controller (pySDC.Controller): The controller
            params (dict): The params passed for this specific convergence controller
            description (dict): The description object used to instantiate the controller

        Returns:
            dict: Updated parameters with default values
        """

        from pySDC.implementations.convergence_controller_classes.hotrod import HotRod
        from pySDC.implementations.convergence_controller_classes.adaptivity import Adaptivity

        default_params = {
            'control_order': -75,
            'use_adaptivity': any([me == Adaptivity for me in description.get('convergence_controllers', {})]),
            'use_HotRod': any([me == HotRod for me in description.get('convergence_controllers', {})]),
            'order_time_marching': description['step_params']['maxiter'],
        }

        new_params = {**default_params, **params}

        # Do a sufficiently high order Taylor expansion
        new_params['Taylor_order'] = new_params['order_time_marching'] + 2

        # Estimate and store values from this iteration
        new_params['estimate_iter'] = new_params['order_time_marching'] - (1 if new_params['use_HotRod'] else 0)

        # Store n values. Since we store u and f, we need only half of each (the +1 is for rounding)
        new_params['n'] = (new_params['Taylor_order'] + 1) // 2
        new_params['n_per_proc'] = new_params['n'] * 1

        return new_params

    def setup_status_variables(self, controller):
        '''
        Initialize coefficient variables.

        Args:
            controller (pySDC.controller): The controller

        Returns:
            None
        '''
        self.coeff.u = [None] * self.params.n
        self.coeff.f = [0.] * self.params.n
        return None

    def check_parameters(self, controller, params, description):
        '''
        Check whether parameters are compatible with whatever assumptions went into the step size functions etc.

        Args:
            controller (pySDC.Controller): The controller
            params (dict): The params passed for this specific convergence controller
            description (dict): The description object used to instantiate the controller

        Returns:
            bool: Whether the parameters are compatible
            str: Error message
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

        Args:
            S (pySDC.Step): The current step

        Returns:
            None
        """
        # figure out which values are to be replaced by the new ones
        if None in self.prev.t:
            oldest_val = len(self.prev.t) - len(self.prev.t[self.prev.t == [None]])
        else:
            oldest_val = np.argmin(self.prev.t)

        # figure out how to store the right hand side
        f = S.levels[0].f[-1]
        if type(f) == imex_mesh:
            self.prev.f[oldest_val] = f.impl + f.expl
        elif type(f) == mesh:
            self.prev.f[oldest_val] = f
        else:
            raise DataError(f'Unable to store f from datatype {type(f)}, extrapolation based error estimate only\
 works with types imex_mesh and mesh')

        # store the rest of the values
        self.prev.u[oldest_val] = S.levels[0].u[-1]
        self.prev.t[oldest_val] = S.time + S.dt
        self.prev.dt[oldest_val] = S.dt

        return None

    def get_extrapolation_coefficients(self, t, dt, t_eval):
        """
        This function solves a linear system where in the matrix A, the row index reflects the order of the derivative
        in the Taylor expansion and the column index reflects the particular step and whether its u or f from that
        step. The vector b on the other hand, contains a 1 in the first entry and zeros elsewhere, since we want to
        compute the value itself and all the derivatives should vanish after combining the Taylor expansions. This
        works to the order of the number of rows and since we want a square matrix for solving, we need the same amount
        of columns, which determines the memory overhead, since it is equal to the solutions / rhs that we need in
        memory at the time of evaluation.

        This is enough to get the extrapolated solution, but if we want to compute the local error, we have to compute
        a prefactor. This is based on error accumulation between steps (first step's solution is exact plus 1 LTE,
        second solution is exact plus 2 LTE and so on), which can be computed for adaptive step sizes as well. However,
        this is only true for linear problems, which means we expect the error estimate to work less well for non-linear
        problems.

        Since only time differences are important for computing the coefficients, we need to compute this only once when
        using constant step sizes. When we allow the step size to change, however, we need to recompute this in every
        step, which is activated by the `use_adaptivity` parameter.

        Solving for the coefficients requires solving a dense linear system of equations. The number of unknowns is
        equal to the order of the Taylor expansion, so this step should be cheap compared to the solves in each SDC
        iteration.

        The function stores the computed coefficients in the `self.coeff` variables.

        Args:
            t (list): The list of times at which we have solutions available
            dt (list): The step sizes used for computing these solutions (needed for the prefactor)
            t_eval (float): The time we want to extrapolate to

        Returns:
            None
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
        self.coeff.u = coeff[:self.params.n]
        self.coeff.f[self.params.n * 2 - self.params.Taylor_order:] = coeff[self.params.n:self.params.Taylor_order]

        # determine prefactor
        step_size_ratios = abs(dt[len(dt) - len(self.coeff.u):] / dt[-1])**(self.params.Taylor_order - 1)
        inv_prefactor = -sum(step_size_ratios[1:]) - 1.
        for i in range(len(self.coeff.u)):
            inv_prefactor += sum(step_size_ratios[1: i + 1]) * self.coeff.u[i]
        self.coeff.prefactor = 1. / abs(inv_prefactor)

        return None


class EstimateExtrapolationErrorNonMPI(EstimateExtrapolationErrorBase):
    '''
    Implementation of the extrapolation error estimate for the non-MPI controller.
    '''

    def setup(self, controller, params, description):
        '''
        Add a no parameter 'no_storage' which decides whether the standart or the no-memory-overhead version is run,
        where only values are used for extrapolation which are in memory of other processes

        Args:
            controller (pySDC.Controller): The controller
            params (dict): The params passed for this specific convergence controller
            description (dict): The description object used to instantiate the controller

        Returns:
            dict: Updated parameters with default values
        '''
        default_params = super(EstimateExtrapolationErrorNonMPI, self).setup(controller, params, description)

        non_mpi_defaults = {
            'no_storage': False,
        }

        return {**non_mpi_defaults, **default_params}

    def setup_status_variables(self, controller):
        '''
        Initialize storage variables.

        Args:
            controller (pySDC.controller): The controller

        Returns:
            None
        '''
        super(EstimateExtrapolationErrorNonMPI, self).setup_status_variables(controller)

        self.prev.t = np.array([None] * self.params.n)
        self.prev.dt = np.array([None] * self.params.n)
        self.prev.u = [None] * self.params.n
        self.prev.f = [None] * self.params.n

        return None

    def post_iteration_processing(self, controller, S):
        '''
        We perform three key operations here in the last iteration:
         - Compute the error estimate
         - Compute the coefficients if needed
         - Store the values of the step if we pretend not to for the no-memory overhead version

        Args:
            controller (pySDC.Controller): The controller
            S (pySDC.Step): The current step

        Returns:
            None
        '''
        if S.status.iter == self.params.estimate_iter:
            t_eval = S.time + S.dt

            # compute the extrapolation coefficients if needed
            if (None in self.coeff.u or self.params.use_adaptivity) and\
                    None not in self.prev.t and t_eval > max(self.prev.t):
                self.get_extrapolation_coefficients(self.prev.t, self.prev.dt, t_eval)

            # compute the error if we can
            if None not in self.coeff.u and None not in self.prev.t:
                self.get_extrapolated_error(S)

            # store the solution and pretend we didn't because in the non MPI version we take a few shortcuts
            if self.params.no_storage:
                self.store_values(S)

        return None

    def prepare_next_block_nonMPI(self, controller, MS, active_slots, time, Tend):
        '''
        If the no-memory-overhead version is used, we need to delete stuff that shouldn't be available. Otherwise, we
        need to store all the stuff that we can.

        Args:
            controller (pySDC.Controller): The controller
            MS (list): All steps of the controller
            active_slots (list): Index list of active steps
            time (float): The current time
            Tend (float): The final time

        Returns:
            None
        '''
        # delete values that should not be available in the next step
        if self.params.no_storage:
            self.prev.t = np.array([None] * self.params.n)
            self.prev.dt = np.array([None] * self.params.n)
            self.prev.u = [None] * self.params.n
            self.prev.f = [None] * self.params.n

        else:
            # decide where we need to restart to store everything up to that point
            MS_active = [MS[i] for i in range(len(MS)) if i in active_slots]
            restarts = [S.status.restart for S in MS_active]
            restart_at = np.where(restarts)[0][0] if True in restarts else len(MS_active)

            # store values in the current block that don't need restarting
            if restart_at > 0:
                [self.store_values(S) for S in MS_active[:restart_at]]

        return None

    def get_extrapolated_solution(self, S):
        '''
        Combine values from previous steps to extrapolate.

        Args:
            S (pySDC.Step): The current step

        Returns:
            dtype_u: The extrapolated solution
        '''
        if len(S.levels) > 1:
            raise NotImplementedError('Extrapolated estimate only works on the finest level for now')

        # prepare variables
        u_ex = S.levels[0].u[-1] * 0.
        idx = np.argsort(self.prev.t)

        # see if we have a solution for the current step already stored
        if (abs(S.time + S.dt - self.prev.t) < 10. * np.finfo(float).eps).any():
            idx_step = idx[np.argmin(abs(self.prev.t - S.time - S.dt))]
        else:
            idx_step = max(idx) + 1

        # make a mask of all the steps we want to include in the extrapolation
        mask = np.logical_and(idx < idx_step, idx >= idx_step - self.params.n)

        # do the extrapolation by summing everything up
        for i in range(self.params.n):
            u_ex += self.coeff.u[i] * self.prev.u[idx[mask][i]] + self.coeff.f[i] * self.prev.f[idx[mask][i]]

        return u_ex

    def get_extrapolated_error(self, S):
        """
        The extrapolation estimate combines values of u and f from multiple steps to extrapolate and compare to the
        solution obtained by the time marching scheme.

        Args:
            S (pySDC.Step): The current step

        Returns:
            None
        """
        u_ex = self.get_extrapolated_solution(S)
        if u_ex is not None:
            S.levels[0].status.error_extrapolation_estimate = abs(u_ex - S.levels[0].u[-1]) * self.coeff.prefactor
        else:
            S.levels[0].status.error_extrapolation_estimate = None

        return None
