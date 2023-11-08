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
            "maxiter": None,
            "use_e_tol": 'e_tol' in description['level_params'].keys(),
            "initial_tol": 1e-3,
            **super().setup(controller, params, description, **kwargs),
        }
        if defaults['maxiter']:
            self.set_maxiter(description, defaults['maxiter'])
        return defaults

    def dependencies(self, controller, description, **kwargs):
        """
        Load the embedded error estimator if needed.

        Args:
            controller (pySDC.Controller): The controller
            description (dict): The description object used to instantiate the controller

        Returns:
            None
        """
        super().dependencies(controller, description)

        if self.params.use_e_tol:
            from pySDC.implementations.convergence_controller_classes.estimate_embedded_error import (
                EstimateEmbeddedError,
            )

            controller.add_convergence_controller(
                EstimateEmbeddedError,
                description=description,
            )

        return None

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
            SDC_accuracy = (
                lvl.status.get('error_embedded_estimate', lvl.status.residual)
                if self.params.use_e_tol
                else lvl.status.residual
            )
            SDC_accuracy = self.params.initial_tol if SDC_accuracy is None else SDC_accuracy
            tol = max([min([SDC_accuracy * self.params.ratio, self.params.max_tol]), self.params.min_tol])
            self.set_tolerance(lvl, tol)
            self.log(f'Changed tolerance to {tol:.2e}', step)

    def set_tolerance(self, lvl, tol):
        lvl.prob.newton_tol = tol

    def set_maxiter(self, description, maxiter):
        description['problem_params']['newton_maxiter'] = maxiter
