import numpy as np
from pySDC.core.convergence_controller import ConvergenceController


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

    def dependencies(self, controller, description, **kwargs):
        """
        Load the slope limiter if needed.

        Args:
            controller (pySDC.Controller): The controller
            description (dict): The description object used to instantiate the controller

        Returns:
            None
        """
        slope_limiter_keys = ['dt_slope_min', 'dt_slope_max', 'dt_rel_min_slope']
        available_keys = [me for me in slope_limiter_keys if me in self.params.__dict__.keys()]

        if len(available_keys) > 0:
            slope_limiter_params = {key: self.params.__dict__[key] for key in available_keys}
            slope_limiter_params['control_order'] = self.params.control_order - 1
            controller.add_convergence_controller(
                StepSizeSlopeLimiter, params=slope_limiter_params, description=description
            )

        return None

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
                        f"Step size exceeds maximum, decreasing from {L.status.dt_new:.2e} to {self.params.dt_max:.2e}",
                        S,
                    )
                    L.status.dt_new = self.params.dt_max

        return None


class StepSizeSlopeLimiter(ConvergenceController):
    """
    Class to set limits to adaptive step size computation during run time

    Please supply `dt_slope_min` or `dt_slope_max` in the params to limit in either direction.
    You can also supply `dt_rel_min_slope` in order to keep the old step size in case the relative change is smaller
    than this minimum.
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
            "control_order": 91,
            "dt_slope_min": 0,
            "dt_slope_max": np.inf,
            "dt_rel_min_slope": 0,
        }
        return {**defaults, **super().setup(controller, params, description, **kwargs)}

    def get_new_step_size(self, controller, S, **kwargs):
        """
        Enforce an upper and lower limit to the slope of the step size here.
        The final step is adjusted such that we reach Tend as best as possible, which might give step sizes below
        the lower limit set here.

        Args:
            controller (pySDC.Controller): The controller
            S (pySDC.Step): The current step

        Returns:
            None
        """
        for L in S.levels:
            if L.status.dt_new is not None:
                if L.status.dt_new / L.params.dt < self.params.dt_slope_min:
                    dt_new = L.params.dt * self.params.dt_slope_min
                    self.log(
                        f"Step size slope is below minimum, increasing from {L.status.dt_new:.2e} to \
{dt_new:.2e}",
                        S,
                    )
                    L.status.dt_new = dt_new
                elif L.status.dt_new / L.params.dt > self.params.dt_slope_max:
                    dt_new = L.params.dt * self.params.dt_slope_max
                    self.log(
                        f"Step size slope exceeds maximum, decreasing from {L.status.dt_new:.2e} to \
{dt_new:.2e}",
                        S,
                    )
                    L.status.dt_new = dt_new
                elif abs(L.status.dt_new / L.params.dt - 1) < self.params.dt_rel_min_slope and not S.status.restart:
                    L.status.dt_new = L.params.dt
                    self.log(
                        f"Step size did not change sufficiently to warrant step size change, keeping {L.status.dt_new:.2e}",
                        S,
                    )

        return None


class StepSizeRounding(ConvergenceController):
    """
    Class to round step size when using adaptive step size selection.
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
            "control_order": +93,
            "digits": 1,
            "fac": 5,
        }
        return {**defaults, **super().setup(controller, params, description, **kwargs)}

    @staticmethod
    def _round_step_size(dt, fac, digits):
        dt_rounded = None
        exponent = np.log10(dt) // 1

        dt_norm = dt / 10 ** (exponent - digits)
        dt_norm_round = (dt_norm // fac) * fac
        dt_rounded = dt_norm_round * 10 ** (exponent - digits)
        return dt_rounded

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
                dt_rounded = self._round_step_size(L.status.dt_new, self.params.fac, self.params.digits)

                if L.status.dt_new != dt_rounded:
                    self.log(
                        f"Step size rounded from {L.status.dt_new:.6e} to {dt_rounded:.6e}",
                        S,
                    )
                    L.status.dt_new = dt_rounded

        return None
