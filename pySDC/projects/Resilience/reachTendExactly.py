import numpy as np
from pySDC.core.convergence_controller import ConvergenceController
from pySDC.implementations.convergence_controller_classes.check_convergence import CheckConvergence


class ReachTendExactly(ConvergenceController):
    """
    This convergence controller will adapt the step size of (hopefully) the last step such that `Tend` is reached very closely.
    Please pass the same `Tend` that you pass to the controller to the params for this to work.
    """

    def setup(self, controller, params, description, **kwargs):
        defaults = {
            "control_order": +94,
            "Tend": None,
            'min_step_size': 1e-10,
        }
        return {**defaults, **super().setup(controller, params, description, **kwargs)}

    def get_new_step_size(self, controller, step, **kwargs):
        L = step.levels[0]
        time_size = step.status.time_size

        if not CheckConvergence.check_convergence(step):
            return None

        dt = L.status.dt_new if L.status.dt_new else L.params.dt
        time_left = self.params.Tend - L.time - L.dt

        if (
            time_left <= (dt + self.params.min_step_size) * time_size
            and not step.status.restart
            and time_left > 0
            and step.status.last
        ):
            dt_new = (
                min([(dt + self.params.min_step_size) * time_size, max([time_left, self.params.min_step_size])])
                + time_size * np.finfo('float').eps * 10
            ) / time_size

            if dt_new != L.status.dt_new:
                L.status.dt_new = dt_new
                self.log(
                    f'Changing step size from {dt:12e} to {L.status.dt_new:.12e} because there is only {time_left:.12e} left.',
                    step,
                )
