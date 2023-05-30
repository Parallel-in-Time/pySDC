from pySDC.core.ConvergenceController import ConvergenceController


class NewtonInexactness(ConvergenceController):
    """
    Gradually refine Newton tolerance based on SDC residual.
    Be aware that the problem needs a parameter called "newton_tol" which controls the tolerance for the Newton solver for this to work!
    """

    def setup(self, controller, params, description, **kwargs):
        """
        Define default parameters here.

        Args:
            controller (pySDC.Controller): The controller
            params (dict): The params passed for this specific convergence controller
            description (dict): The description object used to instantiate the controller

        Returns:
            (dict): The updated params dictionary
        """
        defaults = {
            "control_order": 500,
            "ratio": 1e-2,
            "min_tol": 0,
            "max_tol": 1e99,
        }
        return {**defaults, **super().setup(controller, params, description, **kwargs)}

    def post_iteration_processing(self, controller, step, **kwargs):
        """
        Change the Newton tolerance after every iteration.

        Args:
            controller (pySDC.Controller.controller): The controller
            S (pySDC.Step): The current step

        Returns:
            None
        """
        for lvl in step.levels:
            lvl.prob.newton_tol = max(
                [min([lvl.status.residual * self.params.ratio, self.params.max_tol]), self.params.min_tol]
            )

            self.log(f'Changed Newton tolerance to {lvl.prob.newton_tol:.2e}', step)
