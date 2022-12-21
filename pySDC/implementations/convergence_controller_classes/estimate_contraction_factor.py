import numpy as np

from pySDC.core.ConvergenceController import ConvergenceController
from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
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
        return {"control_order": -75, "e_tol": None, **params}

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
            EstimateEmbeddedError.get_implementation("nonMPI" if type(controller) == controller_nonMPI else "MPI"),
            description=description,
        )

    def setup_status_variables(self, controller, **kwargs):
        """
        Add the embedded error, contraction factor and iterations to convergence variable to the status of the levels.

        Args:
            controller (pySDC.Controller): The controller

        Returns:
            None
        """
        if 'comm' in kwargs.keys():
            steps = [controller.S]
        else:
            if 'active_slots' in kwargs.keys():
                steps = [controller.MS[i] for i in kwargs['active_slots']]
            else:
                steps = controller.MS
        where = ["levels", "status"]
        for S in steps:
            self.add_variable(S, name='error_embedded_estimate_last_iter', where=where, init=None)
            self.add_variable(S, name='contraction_factor', where=where, init=None)
            if self.params.e_tol is not None:
                self.add_variable(S, name='iter_to_convergence', where=where, init=None)

    def reset_status_variables(self, controller, **kwargs):
        """
        Reinitialize new status variables for the levels.

        Args:
            controller (pySDC.controller): The controller

        Returns:
            None
        """
        self.setup_status_variables(controller, **kwargs)

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
