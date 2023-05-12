import numpy as np
from pySDC.core.Hooks import hooks


class LogError(hooks):
    """
    Base class with functions to add the local and global error to the stats, which can be inherited by hooks logging
    these at specific places.

    Errors are computed with respect to `u_exact` defined in the problem class.
    Be aware that this requires the problems to be compatible with this. We need some kind of "exact" solution for this
    to work, be it a reference solution or something analytical.
    """

    def log_global_error(self, step, level_number, suffix=''):
        """
        Function to add the global error to the stats

        Args:
            step (pySDC.Step.step): The current step
            level_number (int): The index of the level
            suffix (str): Suffix for naming the variable in stats

        Returns:
            None
        """
        L = step.levels[level_number]

        L.sweep.compute_end_point()

        u_ref = L.prob.u_exact(t=L.time + L.dt)

        self.add_to_stats(
            process=step.status.slot,
            time=L.time + L.dt,
            level=L.level_index,
            iter=step.status.iter,
            sweep=L.status.sweep,
            type=f'e_global{suffix}',
            value=abs(u_ref - L.uend),
        )
        self.add_to_stats(
            process=step.status.slot,
            time=L.time + L.dt,
            level=L.level_index,
            iter=step.status.iter,
            sweep=L.status.sweep,
            type=f'e_global_rel{suffix}',
            value=abs((u_ref - L.uend / u_ref)),
        )

    def log_local_error(self, step, level_number, suffix=''):
        """
        Function to add the local error to the stats

        Args:
            step (pySDC.Step.step): The current step
            level_number (int): The index of the level
            suffix (str): Suffix for naming the variable in stats

        Returns:
            None
        """
        L = step.levels[level_number]

        L.sweep.compute_end_point()

        self.add_to_stats(
            process=step.status.slot,
            time=L.time + L.dt,
            level=L.level_index,
            iter=step.status.iter,
            sweep=L.status.sweep,
            type=f'e_local{suffix}',
            value=abs(L.prob.u_exact(t=L.time + L.dt, u_init=L.u[0] * 1.0, t_init=L.time) - L.uend),
        )


class LogGlobalErrorPostStep(LogError):
    def post_step(self, step, level_number):
        super().post_step(step, level_number)
        self.log_global_error(step, level_number, '_post_step')


class LogGlobalErrorPostRun(hooks):
    """
    Compute the global error once after the run is finished.
    Because of some timing issues, we cannot inherit from the `LogError` class here.
    The issue is that the convergence controllers can change the step size after the final iteration but before the
    `post_run` functions of the hooks are called, which results in a mismatch of `L.time + L.dt` as corresponding to
    when the solution is computed and when the error is computed. The issue is resolved by recording the time at which
    the solution is computed in a private attribute of this class.

    There is another issue: The MPI controller instantiates a step after the run is completed, meaning the final
    solution is not accessed by computing the end point, but by using the initial value on the finest level.
    Additionally, the number of restarts is reset, which we need to filter recomputed values in post processing.
    For this reason, we need to mess with the private `__num_restarts` of the core Hooks class.
    """

    def __init__(self):
        """
        Add an attribute for when the last solution was added.
        """
        super().__init__()
        self.t_last_solution = 0
        self.num_restarts = 0

    def post_step(self, step, level_number):
        """
        Store the time at which the solution is stored.
        This is required because between the `post_step` hook where the solution is stored and the `post_run` hook
        where the error is stored, the step size can change.

        Args:
            step (pySDC.Step.step): The current step
            level_number (int): The index of the level

        Returns:
            None
        """
        super().post_step(step, level_number)
        self.t_last_solution = step.levels[0].time + step.levels[0].dt
        self.num_restarts = step.status.get('restarts_in_a_row', 0)

    def post_run(self, step, level_number):
        """
        Log the global error.

        Args:
            step (pySDC.Step.step): The current step
            level_number (int): The index of the level

        Returns:
            None
        """
        super().post_run(step, level_number)
        self._hooks__num_restarts = self.num_restarts

        if level_number == 0 and step.status.last:
            L = step.levels[level_number]

            u_num = self.get_final_solution(L)
            u_ref = L.prob.u_exact(t=self.t_last_solution)

            self.logger.info(f'Finished with a global error of e={abs(u_num-u_ref):.2e}')

            self.add_to_stats(
                process=step.status.slot,
                time=self.t_last_solution,
                level=L.level_index,
                iter=step.status.iter,
                sweep=L.status.sweep,
                type='e_global_post_run',
                value=abs(u_num - u_ref),
            )
            self.add_to_stats(
                process=step.status.slot,
                time=self.t_last_solution,
                level=L.level_index,
                iter=step.status.iter,
                sweep=L.status.sweep,
                type='e_global_rel_post_run',
                value=abs((u_num - u_ref) / u_ref),
            )

    def get_final_solution(self, lvl):
        """
        Get the final solution from the level

        Args:
            lvl (pySDC.Level.level): The level
        """
        return lvl.uend


class LogGlobalErrorPostRunMPI(LogGlobalErrorPostRun):
    """
    The MPI controller shows slightly different behaviour which is why the final solution is stored in a different place
    than in the nonMPI controller.
    """

    def post_step(self, step, level_number):
        super().post_step(step, level_number)
        self.num_restarts = self._hooks__num_restarts

    def get_final_solution(self, lvl):
        """
        Get the final solution from the level

        Args:
            lvl (pySDC.Level.level): The level
        """
        return lvl.u[0]


class LogLocalErrorPostStep(LogError):
    """
    Log the local error with respect to `u_exact` defined in the problem class as "e_local_post_step".
    Be aware that this requires the problems to be compatible with this. In particular, a reference solution needs to
    be made available from the initial conditions of the step, not of the run. Otherwise you compute the global error.
    """

    def post_step(self, step, level_number):
        super().post_step(step, level_number)
        self.log_local_error(step, level_number, suffix='_post_step')


class LogLocalErrorPostIter(LogError):
    """
    Log the local error after each iteration
    """

    def post_iteration(self, step, level_number):
        """
        Args:
            step (pySDC.Step.step): the current step
            level_number (int): the current level number

        Returns:
            None
        """
        super().post_iteration(step, level_number)

        self.log_local_error(step, level_number, suffix='_post_iteration')
