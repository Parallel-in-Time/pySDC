from pySDC.core.ConvergenceController import ConvergenceController
from pySDC.implementations.hooks.log_solution import LogToFileAfterXs
import pickle
import numpy as np


class LogStats(ConvergenceController):

    def setup(self, controller, params, *args, **kwargs):
        params['control_order'] = 999
        if 'hook' not in params.keys():
            from pySDC.implementations.hooks.log_solution import LogToFile

            params['hook'] = LogToFile

        self.counter = params['hook'].counter

        return super().setup(controller, params, *args, **kwargs)

    def post_step_processing(self, controller, S, **kwargs):
        hook = self.params.hook
        if self.counter < hook.counter:
            path = f'{hook.path}/{self.params.file_name}_{hook.format_index(hook.counter-1)}.pickle'
            for _hook in controller.hooks:
                _hook.post_step(S, 0)

            stats = controller.return_stats()
            with open(path, 'wb') as file:
                pickle.dump(stats, file)
                self.log(f'Stored stats in {path!r}', S)
            self.counter = hook.counter
