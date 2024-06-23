import numpy as np

from pySDC.core.convergence_controller import ConvergenceController
from pySDC.implementations.convergence_controller_classes.estimate_embedded_error import EstimateEmbeddedError


class EstimateContractionFactor(ConvergenceController):
    """
    Estimate the contraction factor by using the evolution of the embedded error estimate across iterations.
    """

    def setup(self, controller, params, description, **kwargs):
        """
        Add a default value for control order to the parameters.

        Args:
            controller (pySDC.Controller): The controller
            params (dict): Parameters for the convergence controller
            description (dict): The description object used to instantiate the controller

        Returns:
            dict: Updated parameters
        """
        return {"control_order": -75, "e_tol": None, **super().setup(controller, params, description, **kwargs)}

    def dependencies(self, controller, description, **kwargs):
        """
        Load estimator of embedded error.

        Args:
            controller (pySDC.Controller): The controller
            description (dict): The description object used to instantiate the controller

        Returns:
            None
        """
        controller.add_convergence_controller(
            EstimateEmbeddedError,
            description=description,
        )

    def setup_status_variables(self, *args, **kwargs):
        """
        Add the embedded error, contraction factor and iterations to convergence variable to the status of the levels.

        Returns:
            None
        """
        self.add_status_variable_to_level('error_embedded_estimate_last_iter')
        self.add_status_variable_to_level('contraction_factor')
        if self.params.e_tol is not None:
            self.add_status_variable_to_level('iter_to_convergence')

    def post_iteration_processing(self, controller, S, **kwargs):
        """
        Estimate contraction factor here as the ratio of error estimates between iterations and estimate how many more
        iterations we need.

        Args:
            controller (pySDC.controller): The controller
            S (pySDC.step): The current step

        Returns:
            None
        """
        for L in S.levels:
            if L.status.error_embedded_estimate_last_iter is not None:
                L.status.contraction_factor = (
                    L.status.error_embedded_estimate / L.status.error_embedded_estimate_last_iter
                )
                if self.params.e_tol is not None:
                    L.status.iter_to_convergence = max(
                        [
                            0,
                            np.ceil(
                                np.log(self.params.e_tol / L.status.error_embedded_estimate)
                                / np.log(L.status.contraction_factor)
                            ),
                        ]
                    )

    def pre_iteration_processing(self, controller, S, **kwargs):
        """
        Store the embedded error estimate of the current iteration in a different place so it doesn't get overwritten.

        Args:
            controller (pySDC.controller): The controller
            S (pySDC.Step): The current step

        Returns:
            None
        """
        for L in S.levels:
            if L.status.error_embedded_estimate is not None:
                L.status.error_embedded_estimate_last_iter = L.status.error_embedded_estimate * 1.0
