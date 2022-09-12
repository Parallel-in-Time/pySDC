import numpy as np
from pySDC.core.ConvergenceController import ConvergenceController


class SpreadStepSizesBlockwise(ConvergenceController):
    """
    Take the step size from the last step in the block and spread it to all steps in the next block such that every step
    in a block always has the same step size.
    By block we refer to a composite collocation problem, which is solved in pipelined SDC parallel-in-time.

    Also, we overrule the step size control here, if we get close to the final time and we would take too large of a
    step otherwise.
    """

    def setup(self, controller, params, description):
        """
        Define parameters here

        Args:
            controller (pySDC.Controller): The controller
            params (dict): The params passed for this specific convergence controller
            description (dict): The description object used to instantiate the controller

        Returns:
            (dict): The updated params dictionary
        """
        defaults = {
            'control_order': +100,
        }

        return {**defaults, **params}

    def check_parameters(self, controller, params, description):
        """
        Check whether parameters are compatible with whatever assumptions went into the step size functions etc.

        Args:
            controller (pySDC.Controller): The controller
            params (dict): The params passed for this specific convergence controller
            description (dict): The description object used to instantiate the controller

        Returns:
            bool: Whether the parameters are compatible
            str: The error message
        """

        # Have to do this weird try/except block here, because if MPI is not installed, importing the controller will
        # raise an ModuleNotFoundError error. This is not elegant, but the easiest solution so far.
        try:
            from pySDC.implementations.controller_classes.controller_MPI import controller_MPI

            if type(controller) == controller_MPI:
                return False, 'No implementation for spreading step sizes in MPI yet :('

        except ModuleNotFoundError:
            pass

        return True, ''

    def prepare_next_block_nonMPI(self, controller, MS, active_slots, time, Tend):
        """
        Spread the step size of the last step with no restarted predecessors to all steps and limit the step size based
        on Tend

        Args:
            controller (pySDC.Controller): The controller
            MS (list): List of the steps of the controller
            active_slots (list): Index list of active steps
            time (list): List containing the time of all the steps
            Tend (float): Final time of the simulation

        Returns:
            None
        """
        # figure out where the block is restarted
        restarts = [MS[p].status.restart for p in active_slots]
        if True in restarts:
            restart_at = np.where(restarts)[0][0]
        else:
            restart_at = len(restarts) - 1

        # Compute the maximum allowed step size based on Tend.
        dt_max = (Tend - time[0]) / len(active_slots)

        # record the step sizes to restart with from all the levels of the step
        new_steps = [None] * len(MS[restart_at].levels)
        for i in range(len(MS[restart_at].levels)):
            l = MS[restart_at].levels[i]
            # overrule the step size control to reach Tend if needed
            new_steps[i] = min(
                [l.status.dt_new if l.status.dt_new is not None else l.params.dt, max([dt_max, l.params.dt_initial])]
            )

        for p in active_slots:
            # spread the step sizes to all levels
            for i in range(len(MS[p].levels)):
                MS[p].levels[i].params.dt = new_steps[i]

        return None
