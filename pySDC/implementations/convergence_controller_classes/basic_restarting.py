from pySDC.core.ConvergenceController import ConvergenceController, Pars
from pySDC.implementations.convergence_controller_classes.spread_step_sizes import (
    SpreadStepSizesBlockwise,
)
from pySDC.core.Errors import ConvergenceError


class BasicRestarting(ConvergenceController):
    """
    Class with some utilities for restarting. The specific functions are:
     - Telling each step after one that requested a restart to get restarted as well
     - Allowing each step to be restarted a limited number of times in a row before just moving on anyways

    Default control order is 95.
    """

    @classmethod
    def get_implementation(cls, flavor):
        """
        Retrieve the implementation for a specific flavor of this class.

        Args:
            flavor (str): The implementation that you want

        Returns:
            cls: The child class that implements the desired flavor
        """
        if flavor == 'MPI':
            return BasicRestartingMPI
        elif flavor == 'nonMPI':
            return BasicRestartingNonMPI
        else:
            raise NotImplementedError(f'Flavor {flavor} of BasicRestarting is not implemented!')

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
        controller.add_convergence_controller(self.params.step_size_spreader, description=description)
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

    def prepare_next_block(self, controller, S, size, time, Tend, **kwargs):
        """
        Update restarts in a row for all steps.

        Args:
            controller (pySDC.Controller): The controller
            S (pySDC.Step): The current step
            size (int): The number of ranks
            time (list): List containing the time of all the steps
            Tend (float): Final time of the simulation

        Returns:
            None
        """
        S.status.restarts_in_a_row = S.status.restarts_in_a_row + 1 if S.status.restart else 0

        return None


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
        elif not S.status.prev_done:
            # receive information about restarts from earlier ranks
            self.buffers.restart_earlier, self.buffers.max_restart_reached = self.recv(comm, source=S.status.slot - 1)

        # decide whether to restart
        S.status.restart = (S.status.restart or self.buffers.restart_earlier) and not self.buffers.max_restart_reached

        # send information about restarts forward
        if not S.status.last:
            self.send(comm, dest=S.status.slot + 1, data=(S.status.restart, self.buffers.max_restart_reached))

        return None
