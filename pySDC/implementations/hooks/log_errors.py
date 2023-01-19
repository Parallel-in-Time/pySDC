import numpy as np
from pySDC.core.Hooks import hooks


class LogGlobalErrorPostRun(hooks):
    """
    Compute the global error once after the run is finished.
    """

    def __init__(self):
        """
        Add an attribute for when the last solution was added.
        """
        super().__init__()
        self.__t_last_solution = 0

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
        self.__t_last_solution = step.levels[0].time + step.levels[0].dt

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

        if level_number == 0:
            L = step.levels[level_number]

            e_glob = np.linalg.norm(L.prob.u_exact(t=self.__t_last_solution) - L.uend, np.inf)

            if step.status.last:
                self.logger.info(f'Finished with a global error of e={e_glob:.2e}')

            self.add_to_stats(
                process=step.status.slot,
                time=L.time + L.dt,
                level=L.level_index,
                iter=step.status.iter,
                sweep=L.status.sweep,
                type='e_global',
                value=e_glob,
            )
