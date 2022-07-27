import numpy as np
from pySDC.core.ConvergenceController import ConvergenceController
from pySDC.implementations.convergence_controller_classes.estimate_embedded_error import EstimateEmbeddedErrorNonMPI
from pySDC.implementations.convergence_controller_classes.step_size_limiter import StepSizeLimiter
from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI


class Adaptivity(ConvergenceController):
    """
    Class to compute time step size adaptively based on embedded error estimate.
    Adaptivity requires you to know the order of the scheme, which you can also know for Jacobi, but it works
    differently.
    """

    def setup(self, controller, params, description):
        '''
        Define default parameters here

        Args:
            controller (pySDC.Controller): The controller
            params (dict): The params passed for this specific convergence controller
            description (dict): The description object used to instantiate the controller

        Returns:
            (dict): The updated params dictionary
        '''
        return {'control_order': -50, 'beta': 0.9, **params}

    def dependencies(self, controller, description):
        '''
        Load dependencies on other convergence controllers here

        Args:
            controller (pySDC.Controller): The controller
            description (dict): The description object used to instantiate the controller
        '''

        if type(controller) == controller_nonMPI:
            controller.add_convergence_controller(EstimateEmbeddedErrorNonMPI, description=description)
        else:
            raise NotImplementedError('Don\'t have an implementation to estimate the embedded error with MPI')

        if 'dt_min' in self.params.__dict__.keys() or 'dt_max' in self.params.__dict__.keys():
            step_limiter_params = dict()
            step_limiter_params['dt_min'] = self.params.__dict__.get('dt_min', 0)
            step_limiter_params['dt_max'] = self.params.__dict__.get('dt_max', np.inf)
            controller.add_convergence_controller(StepSizeLimiter, params=step_limiter_params, description=description)

    def check_parameters(self, controller, params, description):
        '''
        Check whether parameters are compatible with whatever assumptions went into the step size functions etc.

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
        Determine a step size for the next step from an estimate of the local error of the current step

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
            L.status.dt_new = L.params.dt * self.params.beta * (self.params.e_tol / L.status.error_embedded_estimate)\
                ** (1. / order)
            self.log(f'Adjusting step size from {L.params.dt:.2e} to {L.status.dt_new:.2e}', S)

        return None

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
            if S.levels[0].status.error_embedded_estimate >= self.params.e_tol:
                S.status.restart = True
                self.log(f'Restarting: e={S.levels[0].status.error_embedded_estimate:.2e} >= \
e_tol={self.params.e_tol:.2e}', S)

        return None
