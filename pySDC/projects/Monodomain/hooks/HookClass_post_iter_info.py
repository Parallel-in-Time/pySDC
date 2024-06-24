import time
from pySDC.core.hooks import Hooks


class post_iter_info_hook(Hooks):
    """
    Hook class to write additional iteration information to the command line.
    It is used to print the final residual, after u[0] has been updated with the new value from the previous step.
    This residual is the one used to check the convergence of the iteration and when running in parallel is different from
    the one printed at IT_FINE.
    """

    def __init__(self):
        super(post_iter_info_hook, self).__init__()

    def post_iteration(self, step, level_number):
        """
        Overwrite default routine called after each iteration.
        It calls the default routine and then writes the residual to the command line.
        We call this the residual at IT_END.
        """
        super().post_iteration(step, level_number)
        self.__t1_iteration = time.perf_counter()

        L = step.levels[level_number]

        self.logger.info(
            "Process %2i on time %8.6f at stage %15s: ----------- Iteration: %2i --------------- " "residual: %12.8e",
            step.status.slot,
            L.time,
            "IT_END",
            step.status.iter,
            L.status.residual,
        )
