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
            "spread_from_first_restarted": True,
            "overwrite_to_reach_Tend": True,
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

    def get_step_from_which_to_spread(self, restarts, new_steps, size, S):
        """
        Return the index of the step from which to spread the step size to all steps in the next block.

        Args:
            restarts (list): List of booleans for each step, showing if it wants to be restarted
            new_steps (list): List of the new step sizes on the finest level of each step
            size (int): Size of the time communicator
            S (pySDC.Step.step): The current step

        Returns:
            int: The index of the step from which we want to spread the step size
        """
        if True in restarts:
            restart_at = np.where(restarts)[0][0]
            if self.params.spread_from_first_restarted:
                spread_from_step = restart_at
            else:
                # we want to spread the smallest step size out of the steps that want to be restarted
                spread_from_step = restart_at + np.argmin(new_steps[restart_at:])
            self.debug(f'Detected restart from step {restart_at}. Spreading step size from step {spread_from_step}.', S)
        else:
            restart_at = size - 1
            spread_from_step = restart_at
            self.debug('Spreading step size from last step.', S)

        return spread_from_step, restart_at


class SpreadStepSizesBlockwiseNonMPI(SpreadStepSizesBlockwise):
    """
    Non-MPI version
    """

    def get_step_from_which_to_spread(self, MS, S):
        """
        Return the index of the step from which to spread the step size to all steps in the next block.

        Args:
            MS (list): Active steps
            S (pySDC.Step.step): The current step

        Returns:
            int: The index of the step from which we want to spread the step size
        """
        restarts = [me.status.restart for me in MS]
        new_steps = [me.levels[0].status.dt_new if me.levels[0].status.dt_new else 1e9 for me in MS]
        return super().get_step_from_which_to_spread(restarts, new_steps, len(MS), S)

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

        spread_from_step, restart_at = self.get_step_from_which_to_spread(MS, S)

        # Compute the maximum allowed step size based on Tend.
        dt_all = [0.0] + [me.dt for me in MS if not me.status.first]
        dt_max = (
            (Tend - time[restart_at] - dt_all[restart_at]) / size if self.params.overwrite_to_reach_Tend else np.inf
        )

        # record the step sizes to restart with from all the levels of the step
        new_steps = [None] * len(S.levels)
        for i in range(len(MS[spread_from_step].levels)):
            l = MS[spread_from_step].levels[i]
            # overrule the step size control to reach Tend if needed
            new_steps[i] = min(
                [
                    l.status.dt_new if l.status.dt_new is not None else l.params.dt,
                    max([dt_max, l.params.dt_initial]),
                ]
            )
            if (
                new_steps[i] < (l.status.dt_new if l.status.dt_new is not None else l.params.dt)
                and i == 0
                and l.status.dt_new is not None
            ):
                self.log(
                    f"Overwriting stepsize control to reach Tend: {Tend:.2e}! New step size: {new_steps[i]:.2e}", S
                )

        # spread the step sizes to all levels
        for i in range(len(S.levels)):
            S.levels[i].params.dt = new_steps[i]

        return None


class SpreadStepSizesBlockwiseMPI(SpreadStepSizesBlockwise):
    """
    MPI version
    """

    def get_step_from_which_to_spread(self, comm, S):
        """
        Return the index of the step from which to spread the step size to all steps in the next block.

        Args:
            comm (mpi4py.MPI.Intracomm): Communicator
            S (pySDC.Step.step): The current step

        Returns:
            int: The index of the step from which we want to spread the step size
        """
        restarts = comm.allgather(S.status.restart)
        new_steps = [me if me is not None else 1e9 for me in comm.allgather(S.levels[0].status.dt_new)]

        return super().get_step_from_which_to_spread(restarts, new_steps, comm.size, S)

    def prepare_next_block(self, controller, S, size, time, Tend, comm, **kwargs):
        """
        Spread the step size of the last step with no restarted predecessors to all steps and limit the step size based
        on Tend

        Args:
            controller (pySDC.Controller): The controller
            S (pySDC.step): The current step
            size (int): The number of ranks
            time (float): Time of the first step
            Tend (float): Final time of the simulation
            comm (mpi4py.MPI.Intracomm): Communicator

        Returns:
            None
        """
        spread_from_step, restart_at = self.get_step_from_which_to_spread(comm, S)

        # Compute the maximum allowed step size based on Tend.
        dt_max = comm.bcast((Tend - time) / size, root=restart_at) if self.params.overwrite_to_reach_Tend else np.inf

        # record the step sizes to restart with from all the levels of the step
        new_steps = [None] * len(S.levels)
        if S.status.slot == spread_from_step:
            for i in range(len(S.levels)):
                l = S.levels[i]
                # overrule the step size control to reach Tend if needed
                new_steps[i] = min(
                    [
                        l.status.dt_new if l.status.dt_new is not None else l.params.dt,
                        max([dt_max, l.params.dt_initial]),
                    ]
                )

                if (
                    new_steps[i] < l.status.dt_new
                    if l.status.dt_new is not None
                    else l.params.dt and l.status.dt_new is not None
                ):
                    self.log(
                        f"Overwriting stepsize control to reach Tend: {Tend:.2e}! New step size: {new_steps[i]:.2e}", S
                    )
        new_steps = comm.bcast(new_steps, root=spread_from_step)

        # spread the step sizes to all levels
        for i in range(len(S.levels)):
            S.levels[i].params.dt = new_steps[i]

        return None
