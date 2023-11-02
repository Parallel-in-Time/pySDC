from pySDC.core.ConvergenceController import ConvergenceController
from pySDC.core.Errors import ConvergenceError
import numpy as np


class StopAtNan(ConvergenceController):
    """
    Crash the code when the norm of the solution exceeds some limit or contains nan.
    This class is useful when running with MPI in the sweeper or controller.
    """

    def __init__(self, controller, params, description, **kwargs):
        super().__init__(controller, params, description, **kwargs)
        if self.comm or self.params.useMPI:
            from mpi4py import MPI

            self.MPI_OR = MPI.LOR

    def setup(self, controller, params, description, **kwargs):
        """
        Define parameters here.

        Default parameters are:
         - tresh (float): Crash the code when the norm of the solution exceeds this threshold

        Args:
            controller (pySDC.Controller): The controller
            params (dict): The params passed for this specific convergence controller
            description (dict): The description object used to instantiate the controller

        Returns:
            (dict): The updated params dictionary
        """
        self.comm = description['sweeper_params'].get('comm', None)
        defaults = {
            "control_order": 95,
            "thresh": np.inf,
        }

        return {**defaults, **super().setup(controller, params, description, **kwargs)}

    def post_iteration_processing(self, controller, S, comm=None, **kwargs):
        """
        Check if we need to crash the code.

        Args:
            controller (pySDC.Controller.controller): Controller
            S (pySDC.Step.step): Step
            comm (mpi4py.MPI.Intracomm or None): Communicator of the controller, if applicable

        Raises:
            ConvergenceError: If the solution does not fall within the allowed space
        """
        isfinite, below_limit = True, True
        crash = False

        for lvl in S.levels:
            for u in lvl.u:
                if u is None:
                    break
                isfinite = all(np.isfinite(u))
                below_limit = abs(u) < self.params.thresh

                crash = not (isfinite and below_limit)

                if crash:
                    break
            if crash:
                break

        if self.comm:
            crash = self.comm.allreduce(crash, op=self.MPI_OR)
        elif comm:
            crash = comm.allreduce(crash, op=self.MPI_OR)
        else:
            crash = not isfinite or not below_limit

        if crash:
            raise ConvergenceError(f'Solution exceeds bounds! Crashing code at {S.time}!')
