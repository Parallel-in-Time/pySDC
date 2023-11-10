import numpy as np
import os
import time
from pySDC.core.Hooks import hooks


class post_iter_info_hook(hooks):
    def __init__(self):
        super(post_iter_info_hook, self).__init__()

    def post_iteration(self, step, level_number):
        """
        Default routine called after each iteration

        Args:
            step (pySDC.Step.step): the current step
            level_number (int): the current level number
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
