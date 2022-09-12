import numpy as np
from pySDC.core.ConvergenceController import ConvergenceController


class StepSizeLimiter(ConvergenceController):
    """
    Class to set limits to adaptive step size computation during run time

    Please supply dt_min or dt_max in the params to limit in either direction
    """

    def setup(self, controller, params, description):
        '''
        Define parameters here

        Args:
            controller (pySDC.Controller): The controller
            params (dict): The params passed for this specific convergence controller
            description (dict): The description object used to instantiate the controller

        Returns:
            (dict): The updated params dictionary
        '''
        return {'control_order': +92, 'dt_min': 0, 'dt_max': np.inf, **params}

    def get_new_step_size(self, controller, S):
        '''
        Enforce an upper and lower limit to the step size here

        Args:
            controller (pySDC.Controller): The controller
            S (pySDC.Step): The current step

        Returns:
            None
        '''
        for L in S.levels:
            if L.status.dt_new is not None:
                if L.status.dt_new < self.params.dt_min:
                    self.log(f'Step size is below minimum, increasing from {L.status.dt_new:.2e} to \
{self.params.dt_min:.2e}', S)
                    L.status.dt_new = self.params.dt_min
                elif L.status.dt_new > self.params.dt_max:
                    self.log(f'Step size exceeds maximum, decreasing from {L.status.dt_new:.2e} to \
{self.params.dt_max:.2e}', S)
                    L.status.dt_new = self.params.dt_max

        return None
