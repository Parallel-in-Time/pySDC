import numpy as np
from pySDC.core.ConvergenceController import ConvergenceController, Status
from pySDC.implementations.convergence_controller_classes.store_uold import StoreUOld


class CheckIterationEstimatorNonMPI(ConvergenceController):
    def __init__(self, controller, params, description, **kwargs):
        """
        Initalization routine

        Args:
            controller (pySDC.Controller): The controller
            params (dict): Parameters for the convergence controller
            description (dict): The description object used to instantiate the controller
        """
        super(CheckIterationEstimatorNonMPI, self).__init__(
            controller, params, description
        )
        self.buffers = Status(["Kest_loc", "diff_new", "Ltilde_loc"])
        self.status = Status(["diff_old_loc", "diff_first_loc"])

    def check_parameters(self, controller, params, description, **kwargs):
        """
        Check whether parameters are compatible with whatever assumptions went into the step size functions etc.
        In this case, we need the user to supply a tolerance.

        Args:
            controller (pySDC.Controller): The controller
            params (dict): The params passed for this specific convergence controller
            description (dict): The description object used to instantiate the controller

        Returns:
            bool: Whether the parameters are compatible
            str: The error message
        """
        if "errtol" not in params.keys():
            return (
                False,
                "Please give the iteration estimator a tolerance in the form of `errtol`. Thanks!",
            )

        return True, ""

    def setup(self, controller, params, description, **kwargs):
        """
        Setup parameters. Here we only give a default value for the control order.

        Args:
            controller (pySDC.Controller): The controller
            params (dict): Parameters for the convergence controller
            description (dict): The description object used to instantiate the controller

        Returns:
            dict: The updated parameters
        """
        return {"control_order": -50, **params}

    def dependencies(self, controller, description, **kwargs):
        """
        Need to store the solution of previous iterations.

        Args:
            controller (pySDC.Controller): The controller
            description (dict): The description object used to instantiate the controller

        Returns:
            None
        """
        controller.add_convergence_controller(StoreUOld, description=description)
        return None

    def reset_buffers_nonMPI(self, controller, **kwargs):
        """
        Reset buffers used to immitate communication in non MPI version.

        Args:
            controller (pySDC.controller): The controller

        Returns:
            None
        """
        self.buffers.Kest_loc = [99] * len(controller.MS)
        self.buffers.diff_new = 0.0
        self.buffers.Ltilde_loc = 0.0

    def setup_status_variables(self, controller, **kwargs):
        """
        Setup storage variables for the differences between sweeps for all steps.

        Args:
            controller (pySDC.Controller): The controller

        Returns:
            None
        """
        self.status.diff_old_loc = [0.0] * len(controller.MS)
        self.status.diff_first_loc = [0.0] * len(controller.MS)
        return None

    def check_iteration_status(self, controller, S, **kwargs):
        """
        Args:
            controller (pySDC.Controller): The controller
            S (pySDC.step): The current step

        Returns:
            None
        """
        L = S.levels[0]
        slot = S.status.slot

        # find the global maximum difference between iterations
        for m in range(1, L.sweep.coll.num_nodes + 1):
            self.buffers.diff_new = max(self.buffers.diff_new, abs(L.uold[m] - L.u[m]))

        if S.status.iter == 1:
            self.status.diff_old_loc[slot] = self.buffers.diff_new
            self.status.diff_first_loc[slot] = self.buffers.diff_new
        elif S.status.iter > 1:
            # approximate contraction factor
            self.buffers.Ltilde_loc = min(
                self.buffers.diff_new / self.status.diff_old_loc[slot], 0.9
            )

            self.status.diff_old_loc[slot] = self.buffers.diff_new

            # estimate how many more iterations we need for this step to converge to the desired tolerance
            alpha = 1 / (1 - self.buffers.Ltilde_loc) * self.status.diff_first_loc[slot]
            self.buffers.Kest_loc = (
                np.log(self.params.errtol / alpha)
                / np.log(self.buffers.Ltilde_loc)
                * 1.05
            )
            self.logger.debug(
                f"LOCAL: {L.time:8.4f}, {S.status.iter}: {int(np.ceil(self.buffers.Kest_loc))}, "
                f"{self.buffers.Ltilde_loc:8.6e}, {self.buffers.Kest_loc:8.6e}, \
{self.buffers.Ltilde_loc ** S.status.iter * alpha:8.6e}"
            )

            # set global Kest as last local one, force stop if done
            if S.status.last:
                Kest_glob = self.buffers.Kest_loc
                if np.ceil(Kest_glob) <= S.status.iter:
                    S.status.force_done = True
