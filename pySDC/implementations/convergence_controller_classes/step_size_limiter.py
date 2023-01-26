import numpy as np
from pySDC.core.ConvergenceController import ConvergenceController


class StepSizeLimiter(ConvergenceController):
    """
    Class to set limits to adaptive step size computation during run time

    Please supply dt_min or dt_max in the params to limit in either direction
    """

    def setup(self, controller, params, description, **kwargs):
        """
        Define parameters here

        Args:
            controller (pySDC.Controller): The controller
            params (dict): The params passed for this specific convergence controller
            description (dict): The description object used to instantiate the controller

        Returns:
            (dict): The updated params dictionary
        """
        defaults = {
            "control_order": +92,
            "dt_min": 0,
            "dt_max": np.inf,
        }
        return {**defaults, **super().setup(controller, params, description, **kwargs)}

    def get_new_step_size(self, controller, S, **kwargs):
        """
        Enforce an upper and lower limit to the step size here.
        Be aware that this is only tested when a new step size has been determined. That means if you set an initial
        value for the step size outside of the limits, and you don't do any further step size control, that value will
        go through.
        Also, the final step is adjusted such that we reach Tend as best as possible, which might give step sizes below
        the lower limit set here.

        Args:
            controller (pySDC.Controller): The controller
            S (pySDC.Step): The current step

        Returns:
            None
        """
        for L in S.levels:
            if L.status.dt_new is not None:
                if L.status.dt_new < self.params.dt_min:
                    self.log(
                        f"Step size is below minimum, increasing from {L.status.dt_new:.2e} to \
{self.params.dt_min:.2e}",
                        S,
                    )
                    L.status.dt_new = self.params.dt_min
                elif L.status.dt_new > self.params.dt_max:
                    self.log(
                        f"Step size exceeds maximum, decreasing from {L.status.dt_new:.2e} to \
{self.params.dt_max:.2e}",
                        S,
                    )
                    L.status.dt_new = self.params.dt_max

        return None
