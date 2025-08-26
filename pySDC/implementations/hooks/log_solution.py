from pySDC.core.hooks import Hooks
import pickle
import os
import numpy as np
from pySDC.helpers.fieldsIO import FieldsIO
from pySDC.core.errors import DataError


class LogSolution(Hooks):
    """
    Store the solution at the end of each step as "u".
    """

    def post_step(self, step, level_number):
        """
        Record solution at the end of the step

        Args:
            step (pySDC.Step.step): the current step
            level_number (int): the current level number

        Returns:
            None
        """
        super().post_step(step, level_number)

        L = step.levels[level_number]
        L.sweep.compute_end_point()

        self.add_to_stats(
            process=step.status.slot,
            time=L.time + L.dt,
            level=L.level_index,
            iter=step.status.iter,
            sweep=L.status.sweep,
            type='u',
            value=L.uend,
        )


class LogSolutionAfterIteration(Hooks):
    """
    Store the solution at the end of each iteration as "u".
    """

    def post_iteration(self, step, level_number):
        """
        Record solution at the end of the iteration

        Args:
            step (pySDC.Step.step): the current step
            level_number (int): the current level number

        Returns:
            None
        """
        super().post_iteration(step, level_number)

        L = step.levels[level_number]
        L.sweep.compute_end_point()

        self.add_to_stats(
            process=step.status.slot,
            time=L.time + L.dt,
            level=L.level_index,
            iter=step.status.iter,
            sweep=L.status.sweep,
            type='u',
            value=L.uend,
        )


class LogToPickleFile(Hooks):
    r"""
    Hook for logging the solution to file after the step using pickle.

    Please configure the hook to your liking by manipulating class attributes.
    You must set a custom path to a directory like so:

    ```
    LogToFile.path = '/my/directory/'
    ```

    Keep in mind that the hook will overwrite files without warning!
    You can give a custom file name by setting the ``file_name`` class attribute and give a custom way of rendering the
    index associated with individual files by giving a different function ``format_index`` class attribute. This should
    accept one index and return one string.

    You can also give a custom ``logging_condition`` function, accepting the current level if you want to log selectively.

    Importantly, you may need to change ``process_solution``. By default, this will return a numpy view of the solution.
    Of course, if you are not using numpy, you need to change this. Again, this is a function accepting the level.

    After the fact, you can use the classmethod `get_path` to get the path to a certain data or the `load` function to
    directly load the solution at a given index. Just configure the hook like you did when you recorded the data
    beforehand.

    Finally, be aware that using this hook with MPI parallel runs may lead to different tasks overwriting files. Make
    sure to give a different `file_name` for each task that writes files.
    """

    path = None
    file_name = 'solution'
    counter = 0

    def logging_condition(L):
        return True

    def process_solution(L):
        return {'t': L.time + L.dt, 'u': L.uend.view(np.ndarray)}

    def format_index(index):
        return f'{index:06d}'

    def __init__(self):
        super().__init__()

        if self.path is None:
            raise ValueError('Please set a path for logging as the class attribute `LogToFile.path`!')

        if os.path.isfile(self.path):
            raise ValueError(
                f'{self.path!r} is not a valid path to log to because a file of the same name exists. Please supply a directory'
            )

        if not os.path.isdir(self.path):
            os.makedirs(self.path, exist_ok=True)

    def log_to_file(self, step, level_number, condition, process_solution=None):
        if level_number > 0:
            return None

        L = step.levels[level_number]

        if condition:
            path = self.get_path(self.counter)

            if process_solution:
                data = process_solution(L)
            else:
                data = type(self).process_solution(L)

            with open(path, 'wb') as file:
                pickle.dump(data, file)
            self.logger.info(f'Stored file {path!r}')

            type(self).counter += 1

    def post_step(self, step, level_number):
        L = step.levels[level_number]
        self.log_to_file(step, level_number, type(self).logging_condition(L))

    def pre_run(self, step, level_number):
        L = step.levels[level_number]
        L.uend = L.u[0]

        def process_solution(L):
            return {
                **type(self).process_solution(L),
                't': L.time,
            }

        self.log_to_file(step, level_number, True, process_solution=process_solution)

    @classmethod
    def get_path(cls, index):
        return f'{cls.path}/{cls.file_name}_{cls.format_index(index)}.pickle'

    @classmethod
    def load(cls, index):
        path = cls.get_path(index)
        with open(path, 'rb') as file:
            return pickle.load(file)


class LogToPickleFileAfterXS(LogToPickleFile):
    r'''
    Log to file after certain amount of time has passed instead of after every step
    '''

    time_increment = 0
    t_next_log = 0

    def post_step(self, step, level_number):
        L = step.levels[level_number]

        if self.t_next_log == 0:
            self.t_next_log = self.time_increment

        if L.time + L.dt >= self.t_next_log and not step.status.restart:
            super().post_step(step, level_number)
            self.t_next_log = max([L.time + L.dt, self.t_next_log]) + self.time_increment

    def pre_run(self, step, level_number):
        L = step.levels[level_number]
        L.uend = L.u[0]

        def process_solution(L):
            return {
                **type(self).process_solution(L),
                't': L.time,
            }

        self.log_to_file(step, level_number, type(self).logging_condition(L), process_solution=process_solution)


class LogToFile(Hooks):
    filename = 'myRun.pySDC'
    time_increment = 0
    allow_overwriting = False
    counter = 0  # number of stored time points in the file

    def __init__(self):
        super().__init__()
        self.outfile = None
        self.t_next_log = 0
        FieldsIO.ALLOW_OVERWRITE = self.allow_overwriting

    def pre_run(self, step, level_number):
        if level_number > 0:
            return None
        L = step.levels[level_number]

        # setup outfile
        if os.path.isfile(self.filename) and L.time > 0:
            L.prob.setUpFieldsIO()
            self.outfile = FieldsIO.fromFile(self.filename)
            self.counter = len(self.outfile.times)
            self.logger.info(
                f'Set up file {self.filename!r} for writing output. This file already contains {self.counter} solutions up to t={self.outfile.times[-1]:.4f}.'
            )
        else:
            self.outfile = L.prob.getOutputFile(self.filename)
            self.logger.info(f'Set up file {self.filename!r} for writing output.')

            # write initial conditions
            if L.time not in self.outfile.times:
                self.outfile.addField(time=L.time, field=L.prob.processSolutionForOutput(L.u[0]))
                self.logger.info(f'Written initial conditions at t={L.time:4f} to file')

        type(self).counter = len(self.outfile.times)
        self.logger.info(f'Will write to disk every {self.time_increment:.4e} time units')

    def post_step(self, step, level_number):
        if level_number > 0:
            return None

        L = step.levels[level_number]

        if self.t_next_log == 0:
            self.t_next_log = L.time + self.time_increment

        if L.time + L.dt >= self.t_next_log and not step.status.restart:
            value_exists = True in [abs(me - (L.time + L.dt)) < np.finfo(float).eps * 1000 for me in self.outfile.times]
            if value_exists and not self.allow_overwriting:
                raise DataError(f'Already have recorded data for time {L.time + L.dt} in this file!')
            self.outfile.addField(time=L.time + L.dt, field=L.prob.processSolutionForOutput(L.uend))
            self.logger.info(f'Written solution at t={L.time+L.dt:.4f} to file')
            self.t_next_log = max([L.time + L.dt, self.t_next_log]) + self.time_increment
            type(self).counter = len(self.outfile.times)

    def post_run(self, step, level_number):
        if level_number > 0:
            return None

        L = step.levels[level_number]

        value_exists = True in [abs(me - (L.time + L.dt)) < np.finfo(float).eps * 1000 for me in self.outfile.times]
        if not value_exists:
            self.outfile.addField(time=L.time + L.dt, field=L.prob.processSolutionForOutput(L.uend))
            self.logger.info(f'Written solution at t={L.time+L.dt:.4f} to file')
            self.t_next_log = max([L.time + L.dt, self.t_next_log]) + self.time_increment
            type(self).counter = len(self.outfile.times)

    @classmethod
    def load(cls, index):
        data = {}
        file = FieldsIO.fromFile(cls.filename)
        file_entry = file.readField(idx=index)
        data['u'] = file_entry[1]
        data['t'] = file_entry[0]
        return data
