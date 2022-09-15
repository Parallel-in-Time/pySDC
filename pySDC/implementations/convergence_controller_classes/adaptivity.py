import numpy as np
from pySDC.core.ConvergenceController import ConvergenceController
from pySDC.implementations.convergence_controller_classes.estimate_embedded_error import EstimateEmbeddedErrorNonMPI
from pySDC.implementations.convergence_controller_classes.step_size_limiter import StepSizeLimiter
from pySDC.implementations.convergence_controller_classes.basic_restarting_nonMPI import BasicRestartingNonMPI
from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI


class AdaptivityBase(ConvergenceController):
    """
    Abstract base class for convergence controllers that implement adaptivity based on arbitrary local error estimates
    and update rules.
    """

    def setup(self, controller, params, description):
        '''
        Define default parameters here.

        Default parameters are:
         - control_order (int): The order relative to other convergence controllers
         - beta (float): The safety factor

        Args:
            controller (pySDC.Controller): The controller
            params (dict): The params passed for this specific convergence controller
            description (dict): The description object used to instantiate the controller

        Returns:
            (dict): The updated params dictionary
        '''
        defaults = {
            'control_order': -50,
            'beta': 0.9,
        }
        return {**defaults, **params}

    def dependencies(self, controller, description):
        '''
        Load step size limiters here, if they are desired.

        Args:
            controller (pySDC.Controller): The controller
            description (dict): The description object used to instantiate the controller

        Returns:
            None
        '''

        if 'dt_min' in self.params.__dict__.keys() or 'dt_max' in self.params.__dict__.keys():
            step_limiter_params = dict()
            step_limiter_params['dt_min'] = self.params.__dict__.get('dt_min', 0)
            step_limiter_params['dt_max'] = self.params.__dict__.get('dt_max', np.inf)
            controller.add_convergence_controller(StepSizeLimiter, params=step_limiter_params, description=description)

        return None

    def get_new_step_size(self, controller, S):
        '''
        Determine a step size for the next step from an estimate of the local error of the current step.

        Args:
            controller (pySDC.Controller): The controller
            S (pySDC.Step): The current step

        Returns:
            None
        '''
        raise NotImplementedError('Please implement a rule for updating the step size!')

    def compute_optimal_step_size(self, beta, dt, e_tol, e_est, order):
        '''
        Compute the optimal step size for the current step based on the order of the scheme.
        This function can be called from `get_new_step_size` for various implementations of adaptivity, but notably not
        all! We require to know the order of the error estimate and if we do adaptivity based on the residual, for
        instance, we do not know that and we can't use this function.

        Args:
            beta (float): Safety factor
            dt (float): Current step size
            e_tol (float): The desired tolerance
            e_est (float): The estimated local error
            order (int): The order of the local error estimate

        Returns:
            float: The optimal step size
        '''
        return beta * dt * (e_tol / e_est) ** (1. / order)

    def get_local_error_estimate(self, controller, S, **kwargs):
        '''
        Get the local error estimate for updating the step size.
        It does not have to be an error estimate, but could be the residual or something else.

        Args:
            controller (pySDC.Controller): The controller
            S (pySDC.Step): The current step

        Returns:
            float: The error estimate
        '''
        raise NotImplementedError('Please implement a way to get the local error')

    def determine_restart(self, controller, S):
        '''
        Check if the step wants to be restarted by comparing the estimate of the local error to a preset tolerance

        Args:
            controller (pySDC.Controller): The controller
            S (pySDC.Step): The current step

        Returns:
            None
        '''
        if S.status.iter == S.params.maxiter:
            e_est = self.get_local_error_estimate(controller, S)
            if e_est >= self.params.e_tol:
                S.status.restart = True
                self.log(f'Restarting: e={e_est:.2e} >= e_tol={self.params.e_tol:.2e}', S)

        return None


class Adaptivity(AdaptivityBase):
    """
    Class to compute time step size adaptively based on embedded error estimate.

    We have a version working in non-MPI pipelined SDC, but Adaptivity requires you to know the order of the scheme,
    which you can also know for block-Jacobi, but it works differently and it is only implemented for block
    Gauss-Seidel so far.
    """

    def dependencies(self, controller, description):
        '''
        Load the embedded error estimator.

        Args:
            controller (pySDC.Controller): The controller
            description (dict): The description object used to instantiate the controller

        Returns:
            None
        '''
        super(Adaptivity, self).dependencies(controller, description)
        if type(controller) == controller_nonMPI:
            controller.add_convergence_controller(EstimateEmbeddedErrorNonMPI, description=description)
        else:
            raise NotImplementedError('I only have an implementation of the embedded error for non MPI versions')
        return None

    def check_parameters(self, controller, params, description):
        '''
        Check whether parameters are compatible with whatever assumptions went into the step size functions etc.
        For adaptivity, we need to know the order of the scheme.

        Args:
            controller (pySDC.Controller): The controller
            params (dict): The params passed for this specific convergence controller
            description (dict): The description object used to instantiate the controller

        Returns:
            bool: Whether the parameters are compatible
            str: The error message
        '''
        if description['step_params'].get('restol', -1.) >= 0:
            return False, 'Adaptivity needs constant order in time and hence restol in the step parameters has to be \
smaller than 0!'

        if controller.params.mssdc_jac:
            return False, 'Adaptivity needs the same order on all steps, please activate Gauss-Seidel multistep mode!'

        if 'e_tol' not in params.keys():
            return False, 'Adaptivity needs a local tolerance! Please set some up in description[\'convergence_control\
_params\'][\'e_tol\']!'

        return True, ''

    def get_new_step_size(self, controller, S):
        '''
        Determine a step size for the next step from an embedded estimate of the local error of the current step.

        Args:
            controller (pySDC.Controller): The controller
            S (pySDC.Step): The current step

        Returns:
            None
        '''
        # check if we performed the desired amount of sweeps
        if S.status.iter == S.params.maxiter:
            L = S.levels[0]

            # compute next step size
            order = S.status.iter  # embedded error estimate is same order as time marching
            e_est = self.get_local_error_estimate(controller, S)
            L.status.dt_new = self.compute_optimal_step_size(self.params.beta, L.params.dt, self.params.e_tol, e_est,
                                                             order)
            self.log(f'Adjusting step size from {L.params.dt:.2e} to {L.status.dt_new:.2e}', S)

        return None

    def get_local_error_estimate(self, controller, S, **kwargs):
        '''
        Get the embedded error estimate of the finest level of the step.

        Args:
            controller (pySDC.Controller): The controller
            S (pySDC.Step): The current step

        Returns:
            float: Embedded error estimate
        '''
        return S.levels[0].status.error_embedded_estimate


class AdaptivityRK(Adaptivity):
    '''
    Adaptivity for Runge-Kutta methods. Basically, we need to change the order in the step size update
    '''
    def check_parameters(self, controller, params, description):
        '''
        Check whether parameters are compatible with whatever assumptions went into the step size functions etc.
        For adaptivity, we need to know the order of the scheme.

        Args:
            controller (pySDC.Controller): The controller
            params (dict): The params passed for this specific convergence controller
            description (dict): The description object used to instantiate the controller

        Returns:
            bool: Whether the parameters are compatible
            str: The error message
        '''
        if 'update_order' not in params.keys():
            return False, 'Adaptivity needs an order for the update rule! Please set some up in \
description[\'convergence_control_params\'][\'update_order\']!'

        return super(AdaptivityRK, self).check_parameters(controller, params, description)

    def get_new_step_size(self, controller, S):
        '''
        Determine a step size for the next step from an embedded estimate of the local error of the current step.

        Args:
            controller (pySDC.Controller): The controller
            S (pySDC.Step): The current step

        Returns:
            None
        '''
        # check if we performed the desired amount of sweeps
        if S.status.iter == S.params.maxiter:
            L = S.levels[0]

            # compute next step size
            order = self.params.update_order
            e_est = self.get_local_error_estimate(controller, S)
            L.status.dt_new = self.compute_optimal_step_size(self.params.beta, L.params.dt, self.params.e_tol, e_est,
                                                             order)
            self.log(f'Adjusting step size from {L.params.dt:.2e} to {L.status.dt_new:.2e}', S)

        return None


class AdaptivityResidual(AdaptivityBase):
    '''
    Do adaptivity based on residual.

    Since we don't know a correlation between the residual and the error (for nonlinear problems), we employ a simpler
    rule to update the step size. Instead of giving a local tolerance that we try to hit as closely as possible, we set
    two thresholds for the residual. When we exceed the upper one, we reduce the step size by a factor of 2 and if the
    residual falls below the lower threshold, we double the step size.
    '''

    def setup(self, controller, params, description):
        '''
        Define default parameters here.

        Default parameters are:
         - control_order (int): The order relative to other convergence controllers
         - e_tol_low (float): Lower absolute threshold for the residual
         - e_tol (float): Upper absolute threshold for the residual
         - max_restarts: Override maximum number of restarts

        Args:
            controller (pySDC.Controller): The controller
            params (dict): The params passed for this specific convergence controller
            description (dict): The description object used to instantiate the controller

        Returns:
            (dict): The updated params dictionary
        '''
        defaults = {
            'control_order': -45,
            'e_tol_low': 0,
            'e_tol': np.inf,
            'max_restarts': 2 if 'e_tol_low' in params else None
        }
        return {**defaults, **params}

    def setup_status_variables(self, controller):
        '''
        Change maximum number of allowed restarts here.

        Args:
            controller (pySDC.Controller): The controller

        Reutrns:
            None
        '''
        if self.params.max_restarts is not None:
            conv_controllers = controller.convergence_controllers
            restart_cont = [me for me in conv_controllers if type(me) == BasicRestartingNonMPI]

            if len(restart_cont) == 0:
                raise NotImplementedError("Please implement override of maximum number of restarts!")

            restart_cont[0].params.max_restarts = self.params.max_restarts
        return None

    def check_parameters(self, controller, params, description):
        '''
        Check whether parameters are compatible with whatever assumptions went into the step size functions etc.
        For adaptivity, we want a fixed order of the scheme.

        Args:
            controller (pySDC.Controller): The controller
            params (dict): The params passed for this specific convergence controller
            description (dict): The description object used to instantiate the controller

        Returns:
            bool: Whether the parameters are compatible
            str: The error message
        '''
        if description['step_params'].get('restol', -1.) >= 0:
            return False, 'Adaptivity needs constant order in time and hence restol in the step parameters has to be \
smaller than 0!'

        if controller.params.mssdc_jac:
            return False, 'Adaptivity needs the same order on all steps, please activate Gauss-Seidel multistep mode!'

        return True, ''

    def get_new_step_size(self, controller, S):
        '''
        Determine a step size for the next step.
        If we exceed the absolute tolerance of the residual in either direction, we either double or halve the step
        size.

        Args:
            controller (pySDC.Controller): The controller
            S (pySDC.Step): The current step

        Returns:
            None
        '''
        # check if we performed the desired amount of sweeps
        if S.status.iter == S.params.maxiter:
            L = S.levels[0]

            res = self.get_local_error_estimate(controller, S)

            dt_planned = L.status.dt_new if L.status.dt_new is not None else L.params.dt

            if res > self.params.e_tol:
                L.status.dt_new = min([dt_planned, L.params.dt / 2.])
                self.log(f'Adjusting step size from {L.params.dt:.2e} to {L.status.dt_new:.2e}', S)
            elif res < self.params.e_tol_low:
                L.status.dt_new = max([dt_planned, L.params.dt * 2.])
                self.log(f'Adjusting step size from {L.params.dt:.2e} to {L.status.dt_new:.2e}', S)

        return None

    def get_local_error_estimate(self, controller, S, **kwargs):
        '''
        Get the residual of the finest level of the step.

        Args:
            controller (pySDC.Controller): The controller
            S (pySDC.Step): The current step

        Returns:
            float: Embedded error estimate
        '''
        return S.levels[0].status.residual
