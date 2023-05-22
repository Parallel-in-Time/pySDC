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

    def setup(self, controller, params, description, **kwargs):
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
            "control_order": +100,
        }

        return {**defaults, **super().setup(controller, params, description, **kwargs)}

    @classmethod
    def get_implementation(cls, useMPI, **kwargs):
        """
        Get MPI or non-MPI version

        Args:
            useMPI (bool): The implementation that you want

        Returns:
            cls: The child class implementing the desired flavor
        """
        if useMPI:
            return SpreadStepSizesBlockwiseMPI
        else:
            return SpreadStepSizesBlockwiseNonMPI


class SpreadStepSizesBlockwiseNonMPI(SpreadStepSizesBlockwise):
    """
    Non-MPI version
    """

    def prepare_next_block(self, controller, S, size, time, Tend, MS, **kwargs):
        """
        Spread the step size of the last step with no restarted predecessors to all steps and limit the step size based
        on Tend

        Args:
            controller (pySDC.Controller): The controller
            S (pySDC.step): The current step
            size (int): The number of ranks
            time (list): List containing the time of all the steps handled by the controller (or float in MPI implementation)
            Tend (float): Final time of the simulation
            MS (list): Active steps

        Returns:
            None
        """
        # inactive steps don't need to participate
        if S not in MS:
            return None

        # figure out where the block is restarted
        restarts = [me.status.restart for me in MS]
        if True in restarts:
            restart_at = np.where(restarts)[0][0]
        else:
            restart_at = len(restarts) - 1

        # Compute the maximum allowed step size based on Tend.
        dt_max = (Tend - time[0]) / size

        # record the step sizes to restart with from all the levels of the step
        new_steps = [None] * len(S.levels)
        for i in range(len(MS[restart_at].levels)):
            l = MS[restart_at].levels[i]
            # overrule the step size control to reach Tend if needed
            new_steps[i] = min(
                [
                    l.status.dt_new if l.status.dt_new is not None else l.params.dt,
                    max([dt_max, l.params.dt_initial]),
                ]
            )

            if new_steps[i] < (l.status.dt_new if l.status.dt_new is not None else l.params.dt) and i == 0:
                self.log(
                    f"Overwriting stepsize control to reach Tend: {Tend:.2e}! New step size: {new_steps[i]:.2e}", S
                )

        # spread the step sizes to all levels
        for i in range(len(S.levels)):
            S.levels[i].params.dt = new_steps[i]

        return None


class SpreadStepSizesBlockwiseMPI(SpreadStepSizesBlockwise):
    def prepare_next_block(self, controller, S, size, time, Tend, comm, **kwargs):
        """
        Spread the step size of the last step with no restarted predecessors to all steps and limit the step size based
        on Tend

        Args:
            controller (pySDC.Controller): The controller
            S (pySDC.step): The current step
            size (int): The number of ranks
            time (list): List containing the time of all the steps handled by the controller (or float in MPI implementation)
            Tend (float): Final time of the simulation
            comm (mpi4py.MPI.Intracomm): Communicator

        Returns:
            None
        """

        # figure out where the block is restarted
        restarts = comm.allgather(S.status.restart)
        if True in restarts:
            restart_at = np.where(restarts)[0][0]
        else:
            restart_at = len(restarts) - 1

        # Compute the maximum allowed step size based on Tend.
        dt_max = comm.bcast((Tend - time) / size, root=restart_at)

        # record the step sizes to restart with from all the levels of the step
        new_steps = [None] * len(S.levels)
        if S.status.slot == restart_at:
            for i in range(len(S.levels)):
                l = S.levels[i]
                # overrule the step size control to reach Tend if needed
                new_steps[i] = min(
                    [
                        l.status.dt_new if l.status.dt_new is not None else l.params.dt,
                        max([dt_max, l.params.dt_initial]),
                    ]
                )

                if new_steps[i] < l.status.dt_new if l.status.dt_new is not None else l.params.dt:
                    self.log(
                        f"Overwriting stepsize control to reach Tend: {Tend:.2e}! New step size: {new_steps[i]:.2e}", S
                    )
        new_steps = comm.bcast(new_steps, root=restart_at)

        # spread the step sizes to all levels
        for i in range(len(S.levels)):
            S.levels[i].params.dt = new_steps[i]

        return None
