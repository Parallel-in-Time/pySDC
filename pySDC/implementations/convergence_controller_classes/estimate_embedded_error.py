import numpy as np

from pySDC.core.ConvergenceController import ConvergenceController, Pars
from pySDC.implementations.convergence_controller_classes.store_uold import StoreUOld


class EstimateEmbeddedError(ConvergenceController):
    """
    The embedded error is obtained by computing two solutions of different accuracy and pretending the more accurate
    one is an exact solution from the point of view of the less accurate solution. In practice, we like to compute the
    solutions with different order methods, meaning that in SDC we can just subtract two consecutive sweeps, as long as
    you make sure your preconditioner is compatible, which you have to just try out...
    """

    def setup(self, controller, params, description):
        """
        Add a default value for control order to the parameters

        Args:
            controller (pySDC.Controller): The controller
            params (dict): Parameters for the convergence controller
            description (dict): The description object used to instantiate the controller

        Returns:
            dict: Updated parameters
        """
        return {'control_order': -80, **params}

    def dependencies(self, controller, description):
        """
        Load the convergence controller that stores the solution of the last sweep

        Args:
            controller (pySDC.Controller): The controller
            description (dict): The description object used to instantiate the controller

        Returns:
            None
        """
        controller.add_convergence_controller(StoreUOld, description=description)
        return None


class EstimateEmbeddedErrorNonMPI(EstimateEmbeddedError):
    def __init__(self, controller, params, description):
        """
        Initalization routine. Add the buffers for communication over the parent class.

        Args:
            controller (pySDC.Controller): The controller
            params (dict): Parameters for the convergence controller
            description (dict): The description object used to instantiate the controller
        """
        super(EstimateEmbeddedErrorNonMPI, self).__init__(controller, params, description)
        self.buffers = Pars({'e_em_last': 0.0})

    def reset_buffers_nonMPI(self, controller):
        """
        Reset buffers for immitated communication.

        Args:
            controller (pySDC.controller): The controller

        Returns:
            None
        """
        self.buffers.e_em_last = 0.0
        return None

    def post_iteration_processing(self, controller, S):
        """
        Compute embedded error estimate on the last node of each level
        In serial this is the local error, but in block Gauss-Seidel MSSDC this is a semi-global error in each block

        Args:
            controller (pySDC.Controller): The controller
            S (pySDC.Step): The current step

        Returns:
            None
        """
        if len(S.levels) > 1 and len(controller.MS) > 1:
            raise NotImplementedError(
                'Embedded error estimate only works for serial multi-level or parallel single \
level'
            )

        if S.status.iter > 1:
            for L in S.levels:
                # order rises by one between sweeps, making this so ridiculously easy
                temp = abs(L.uold[-1] - L.u[-1])
                L.status.error_embedded_estimate = max([abs(temp - self.buffers.e_em_last), np.finfo(float).eps])

            self.buffers.e_em_last = temp * 1.0

        return None
