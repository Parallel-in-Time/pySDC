from pySDC.core.ConvergenceController import ConvergenceController, Pars
from pySDC.implementations.convergence_controller_classes.spread_step_sizes import (
    SpreadStepSizesBlockwise,
)
from pySDC.core.Errors import ConvergenceError
import numpy as np


class BasicRestarting(ConvergenceController):
    """
    Class with some utilities for restarting. The specific functions are:
     - Telling each step after one that requested a restart to get restarted as well
     - Allowing each step to be restarted a limited number of times in a row before just moving on anyways

    Default control order is 95.
    """

    @classmethod
    def get_implementation(cls, useMPI):
        """
        Retrieve the implementation for a specific flavor of this class.

        Args:
            useMPI (bool): Whether or not the controller uses MPI

        Returns:
            cls: The child class that implements the desired flavor
        """
        if useMPI:
            return BasicRestartingMPI
        else:
            return BasicRestartingNonMPI

    def __init__(self, controller, params, description, **kwargs):
        """
        Initialization routine

        Args:
            controller (pySDC.Controller): The controller
            params (dict): Parameters for the convergence controller
            description (dict): The description object used to instantiate the controller
        """
        super().__init__(controller, params, description)
        self.buffers = Pars({"restart": False, "max_restart_reached": False})

    def setup(self, controller, params, description, **kwargs):
        """
        Define parameters here.

        Default parameters are:
         - control_order (int): The order relative to other convergence controllers
         - max_restarts (int): Maximum number of restarts we allow each step before we just move on with whatever we
                               have
         - step_size_spreader (pySDC.ConvergenceController): A convergence controller that takes care of distributing
                                                             the steps sizes between blocks

        Args:
            controller (pySDC.Controller): The controller
            params (dict): The params passed for this specific convergence controller
            description (dict): The description object used to instantiate the controller

        Returns:
            (dict): The updated params dictionary
        """
        defaults = {
            "control_order": 95,
            "max_restarts": 10,
            "crash_after_max_restarts": True,
            "restart_from_first_step": True,
            "step_size_spreader": SpreadStepSizesBlockwise.get_implementation(useMPI=params['useMPI']),
        }

        from pySDC.implementations.hooks.log_restarts import LogRestarts

        controller.add_hook(LogRestarts)

        return {**defaults, **super().setup(controller, params, description, **kwargs)}

    def setup_status_variables(self, controller, **kwargs):
        """
        Add status variables for whether to restart now and how many times the step has been restarted in a row to the
        Steps

        Args:
            controller (pySDC.Controller): The controller
            reset (bool): Whether the function is called for the first time or to reset

        Returns:
            None
        """
        where = ["S" if 'comm' in kwargs.keys() else "MS", "status"]
        self.add_variable(controller, name='restart', where=where, init=False)
        self.add_variable(controller, name='restarts_in_a_row', where=where, init=0)

    def reset_status_variables(self, controller, reset=False, **kwargs):
        """
        Add status variables for whether to restart now and how many times the step has been restarted in a row to the
        Steps

        Args:
            controller (pySDC.Controller): The controller
            reset (bool): Whether the function is called for the first time or to reset

        Returns:
            None
        """
        where = ["S" if 'comm' in kwargs.keys() else "MS", "status"]
        self.reset_variable(controller, name='restart', where=where, init=False)

    def dependencies(self, controller, description, **kwargs):
        """
        Load a convergence controller that spreads the step sizes between steps.

        Args:
            controller (pySDC.Controller): The controller
            description (dict): The description object used to instantiate the controller

        Returns:
            None
        """
        spread_step_sizes_params = {
            'spread_from_first_restarted': not self.params.restart_from_first_step,
        }
        controller.add_convergence_controller(
            self.params.step_size_spreader, description=description, params=spread_step_sizes_params
        )
        return None

    def determine_restart(self, controller, S, **kwargs):
        """
        Restart all steps after the first one which wants to be restarted as well, but also check if we lost patience
        with the restarts and want to move on anyways.

        Args:
            controller (pySDC.Controller): The controller
            S (pySDC.Step): The current step

        Returns:
            None
        """
        raise NotImplementedError("Please implement a function to determine if we need a restart here!")


class BasicRestartingNonMPI(BasicRestarting):
    """
    Non-MPI specific version of basic restarting
    """

    def reset_buffers_nonMPI(self, controller, **kwargs):
        """
        Reset all variables with are used to simulate communication here

        Args:
            controller (pySDC.Controller): The controller

        Returns:
            None
        """
        self.buffers.restart = False
        self.buffers.max_restart_reached = False

        return None

    def determine_restart(self, controller, S, MS, **kwargs):
        """
        Restart all steps after the first one which wants to be restarted as well, but also check if we lost patience
        with the restarts and want to move on anyways.

        Args:
            controller (pySDC.Controller): The controller
            S (pySDC.Step): The current step
            MS (list): List of active steps

        Returns:
            None
        """
        # check if we performed too many restarts
        if S.status.first:
            self.buffers.max_restart_reached = S.status.restarts_in_a_row >= self.params.max_restarts

            if self.buffers.max_restart_reached and S.status.restart:
                if self.params.crash_after_max_restarts:
                    raise ConvergenceError(f"Restarted {S.status.restarts_in_a_row} time(s) already, surrendering now.")
                self.log(
                    f"Step(s) restarted {S.status.restarts_in_a_row} time(s) already, maximum reached, moving \
on...",
                    S,
                )

        self.buffers.restart = S.status.restart or self.buffers.restart
        S.status.restart = (S.status.restart or self.buffers.restart) and not self.buffers.max_restart_reached

        if S.status.last and self.params.restart_from_first_step and not self.buffers.max_restart_reached:
            for step in MS:
                step.status.restart = self.buffers.restart

        return None

    def prepare_next_block(self, controller, S, size, time, Tend, MS, **kwargs):
        """
        Update restarts in a row for all steps.

        Args:
            controller (pySDC.Controller): The controller
            S (pySDC.Step): The current step
            size (int): The number of ranks
            time (list): List containing the time of all the steps
            Tend (float): Final time of the simulation
            MS (list): List of active steps

        Returns:
            None
        """
        if S not in MS:
            return None

        restart_from = min([me.status.slot for me in MS if me.status.restart] + [size - 1])

        if S.status.slot < restart_from:
            MS[restart_from - S.status.slot].status.restarts_in_a_row = 0
        else:
            step = MS[S.status.slot - restart_from]
            step.status.restarts_in_a_row = S.status.restarts_in_a_row + 1 if S.status.restart else 0

        return None


class BasicRestartingMPI(BasicRestarting):
    """
    MPI specific version of basic restarting
    """

    def __init__(self, controller, params, description, **kwargs):
        """
        Initialization routine. Adds a buffer.

        Args:
            controller (pySDC.Controller): The controller
            params (dict): Parameters for the convergence controller
            description (dict): The description object used to instantiate the controller
        """
        from mpi4py import MPI

        self.OR = MPI.LOR
        self.INT = MPI.INT

        super().__init__(controller, params, description)
        self.buffers = Pars({"restart": False, "max_restart_reached": False, 'restart_earlier': False})

    def determine_restart(self, controller, S, comm, **kwargs):
        """
        Restart all steps after the first one which wants to be restarted as well, but also check if we lost patience
        with the restarts and want to move on anyways.

        Args:
            controller (pySDC.Controller): The controller
            S (pySDC.Step): The current step
            comm (mpi4py.MPI.Intracomm): Communicator

        Returns:
            None
        """
        assert S.status.slot == comm.rank

        if S.status.first:
            # check if we performed too many restarts
            self.buffers.max_restart_reached = S.status.restarts_in_a_row >= self.params.max_restarts
            self.buffers.restart_earlier = False  # there is no earlier step

            if self.buffers.max_restart_reached and S.status.restart:
                if self.params.crash_after_max_restarts:
                    raise ConvergenceError(f"Restarted {S.status.restarts_in_a_row} time(s) already, surrendering now.")
                self.log(
                    f"Step(s) restarted {S.status.restarts_in_a_row} time(s) already, maximum reached, moving \
on...",
                    S,
                )
        elif not S.status.prev_done and not self.params.restart_from_first_step:
            # receive information about restarts from earlier ranks
            self.buffers.restart_earlier, self.buffers.max_restart_reached = self.recv(comm, source=S.status.slot - 1)

        # decide whether to restart
        S.status.restart = (S.status.restart or self.buffers.restart_earlier) and not self.buffers.max_restart_reached

        # send information about restarts forward
        if not S.status.last and not self.params.restart_from_first_step:
            self.send(comm, dest=S.status.slot + 1, data=(S.status.restart, self.buffers.max_restart_reached))

        if self.params.restart_from_first_step:
            max_restart_reached = comm.bcast(S.status.restarts_in_a_row > self.params.max_restarts, root=0)
            S.status.restart = comm.allreduce(S.status.restart, op=self.OR) and not max_restart_reached

        return None

    def prepare_next_block(self, controller, S, size, time, Tend, comm, **kwargs):
        """
        Update restarts in a row for all steps.

        Args:
            controller (pySDC.Controller): The controller
            S (pySDC.Step): The current step
            size (int): The number of ranks
            time (list): List containing the time of all the steps
            Tend (float): Final time of the simulation
            comm (mpi4py.MPI.Intracomm): Communicator

        Returns:
            None
        """

        restart_from = min(comm.allgather(S.status.slot if S.status.restart else S.status.time_size - 1))

        # send "backward" the number of restarts in a row
        if S.status.slot >= restart_from:
            buff = np.empty(1, dtype=int)
            buff[0] = int(S.status.restarts_in_a_row + 1 if S.status.restart else 0)
            self.Send(
                comm,
                dest=S.status.slot - restart_from,
                buffer=[buff, self.INT],
                blocking=False,
            )

        # receive new number of restarts in a row
        if S.status.slot + restart_from < size:
            buff = np.empty(1, dtype=int)
            self.Recv(comm, source=(S.status.slot + restart_from), buffer=[buff, self.INT])
            S.status.restarts_in_a_row = buff[0]
        else:
            S.status.restarts_in_a_row = 0

        return None
