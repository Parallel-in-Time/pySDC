import numpy as np

from pySDC.core.ConvergenceController import ConvergenceController
from pySDC.implementations.convergence_controller_classes.estimate_extrapolation_error import (
    EstimateExtrapolationErrorNonMPI,
)


class HotRod(ConvergenceController):
    """
    Class that incorporates the Hot Rod detector [1] for soft faults. Based on comparing two estimates of the local
    error.

    Default control order is -40.

    See for the reference:
    [1]: Lightweight and Accurate Silent Data Corruption Detection in Ordinary Differential Equation Solvers,
    Guhur et al. 2016, Springer. DOI: https://doi.org/10.1007/978-3-319-43659-3_47
    """

    def setup(self, controller, params, description, **kwargs):
        """
        Setup default values for crucial parameters.

        Args:
            controller (pySDC.Controller): The controller
            params (dict): The params passed for this specific convergence controller
            description (dict): The description object used to instantiate the controller

        Returns:
            dict: The updated params
        """
        default_params = {
            "HotRod_tol": np.inf,
            "control_order": -40,
            "no_storage": False,
        }
        return {**default_params, **super().setup(controller, params, description, **kwargs)}

    def dependencies(self, controller, description, **kwargs):
        """
        Load the dependencies of Hot Rod, which are the two error estimators

        Args:
            controller (pySDC.Controller): The controller
            description (dict): The description object used to instantiate the controller

        Returns:
            None
        """
        from pySDC.implementations.convergence_controller_classes.estimate_embedded_error import EstimateEmbeddedError

        controller.add_convergence_controller(
            EstimateEmbeddedError.get_implementation(flavor='linearized', useMPI=self.params.useMPI),
            description=description,
        )
        if not self.params.useMPI:
            controller.add_convergence_controller(
                EstimateExtrapolationErrorNonMPI, description=description, params={'no_storage': self.params.no_storage}
            )
        else:
            raise NotImplementedError("Don't know how to estimate extrapolated error with MPI")

    def check_parameters(self, controller, params, description, **kwargs):
        """
        Check whether parameters are compatible with whatever assumptions went into the step size functions etc.

        Args:
            controller (pySDC.Controller): The controller
            params (dict): The params passed for this specific convergence controller
            description (dict): The description object used to instantiate the controller

        Returns:
            bool: Whether the parameters are compatible
            str: Error message
        """
        if self.params.HotRod_tol == np.inf:
            controller.logger.warning(
                "Hot Rod needs a detection threshold, which is now set to infinity, such that a \
restart is never triggered!"
            )

        if description["step_params"].get("restol", -1.0) >= 0:
            return (
                False,
                "Hot Rod needs constant order in time and hence restol in the step parameters has to be \
smaller than 0!",
            )

        if controller.params.mssdc_jac:
            return (
                False,
                "Hot Rod needs the same order on all steps, please activate Gauss-Seidel multistep mode!",
            )

        return True, ""

    def determine_restart(self, controller, S, **kwargs):
        """
        Check if the difference between the error estimates exceeds the allowed tolerance

        Args:
            controller (pySDC.Controller): The controller
            S (pySDC.Step): The current step

        Returns:
            None
        """
        # we determine whether to restart only on the last sweep
        if S.status.iter < S.params.maxiter:
            return None

        for L in S.levels:
            if None not in [
                L.status.error_extrapolation_estimate,
                L.status.error_embedded_estimate,
            ]:
                diff = abs(L.status.error_extrapolation_estimate - L.status.error_embedded_estimate)
                if diff > self.params.HotRod_tol:
                    S.status.restart = True
                    self.log(
                        f"Triggering restart: delta={diff:.2e}, tol={self.params.HotRod_tol:.2e}",
                        S,
                    )

        return None

    def post_iteration_processing(self, controller, S, **kwargs):
        """
        Throw away the final sweep to match the error estimates.

        Args:
            controller (pySDC.Controller): The controller
            S (pySDC.Step): The current step

        Returns:
            None
        """
        if S.status.iter == S.params.maxiter:
            for L in S.levels:
                L.u[:] = L.uold[:]

        return None
