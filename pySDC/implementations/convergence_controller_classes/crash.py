from pySDC.core.ConvergenceController import ConvergenceController
from pySDC.core.Errors import ConvergenceError
import numpy as np
import time


class CrashBase(ConvergenceController):
    """
    Crash the code across all ranks
    """

    def __init__(self, controller, params, description, **kwargs):
        super().__init__(controller, params, description, **kwargs)
        if self.comm or self.params.useMPI:
            from mpi4py import MPI

            self.MPI_OR = MPI.LOR

    def communicate_crash(self, crash, msg='', comm=None, **kwargs):
        """
        Communicate a crash across all ranks and raise an error if so.

        Args:
            crash (bool): If this rank wants to crash
            comm (mpi4py.MPI.Intracomm or None): Communicator of the controller, if applicable:
        """

        # communicate across the sweeper
        if self.comm:
            crash = self.comm.allreduce(crash, op=self.MPI_OR)

        # communicate across the steps
        if comm:
            crash = comm.allreduce(crash, op=self.MPI_OR)

        if crash:
            raise ConvergenceError(msg)


class StopAtNan(CrashBase):
    """
    Crash the code when the norm of the solution exceeds some limit or contains nan.
    This class is useful when running with MPI in the sweeper or controller.
    """

    def setup(self, controller, params, description, **kwargs):
        """
        Define parameters here.

        Default parameters are:
         - thresh (float): Crash the code when the norm of the solution exceeds this threshold

        Args:
            controller (pySDC.Controller): The controller
            params (dict): The params passed for this specific convergence controller
            description (dict): The description object used to instantiate the controller

        Returns:
            (dict): The updated params dictionary
        """
        self.comm = description['sweeper_params'].get('comm', None)
        defaults = {
            "control_order": 94,
            "thresh": np.inf,
        }

        return {**defaults, **super().setup(controller, params, description, **kwargs)}

    def prepare_next_block(self, controller, S, *args, **kwargs):
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
                isfinite = np.all(np.isfinite(u))

                below_limit = abs(u) < self.params.thresh

                crash = not (isfinite and below_limit)

                if crash:
                    break
            if crash:
                break

        self.communicate_crash(crash, msg=f'Solution exceeds bounds! Crashing code at {S.time}!', **kwargs)


class StopAtMaxRuntime(CrashBase):
    """
    Abort the code when the problem has exceeded a maximum runtime.
    """

    def setup(self, controller, params, description, **kwargs):
        """
        Define parameters here.

        Default parameters are:
         - max_runtime (float): Crash the code when the norm of the runtime exceeds this threshold

        Args:
            controller (pySDC.Controller): The controller
            params (dict): The params passed for this specific convergence controller
            description (dict): The description object used to instantiate the controller

        Returns:
            (dict): The updated params dictionary
        """
        self.comm = description['sweeper_params'].get('comm', None)
        defaults = {
            "control_order": 94,
            "max_runtime": np.inf,
        }
        self.t0 = time.perf_counter()

        return {**defaults, **super().setup(controller, params, description, **kwargs)}

    def prepare_next_block(self, controller, S, *args, **kwargs):
        """
        Check if we need to crash the code.

        Args:
            controller (pySDC.Controller.controller): Controller
            S (pySDC.Step.step): Step
            comm (mpi4py.MPI.Intracomm or None): Communicator of the controller, if applicable

        Raises:
            ConvergenceError: If the solution does not fall within the allowed space
        """
        self.communicate_crash(
            crash=abs(self.t0 - time.perf_counter()) > self.params.max_runtime,
            msg=f'Exceeding max. runtime of {self.params.max_runtime}s! Crashing code at {S.time}!',
            **kwargs,
        )
