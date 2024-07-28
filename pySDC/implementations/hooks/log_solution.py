from pySDC.core.hooks import Hooks
import pickle
import os
import numpy as np


class LogSolution(Hooks):
    """
    Store the solution at the end of each step as "u". It is also possible to get the solution
    on each node as "u_dense" and corresponding collocation nodes as "nodes" at the end
    of the step.
    """

    def post_step(self, step, level_number):
        """
        Record solution at the end of the step

        Parameters
        ----------
        step : pySDC.core.step.Step
            Current step.
        level_number : int
            Current level number.

        Returns
        -------
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

        if hasattr(L.sweep, 'comm'):
            comm = L.sweep.comm
            size = comm.Get_size()

            uTmp = L.u if not L.sweep.coll.left_is_node else L.u[1:]

            # Determine the shape of arrays to handle None values
            shape = max((me.shape for me in uTmp if me is not None), default=(0,))
            placeholder = np.zeros(shape)

            # Replace None values with placeholder
            u = [me if me is not None else placeholder for me in uTmp]

            # Flatten the list to a single array for Allgather
            uFlat = np.concatenate(u)

            # Prepare the buffer to receive data from all processes
            recvBuf = np.empty(size * len(uFlat), dtype='d')

            # Use Allgather to collect data from all processes
            comm.Allgather(uFlat, recvBuf)

            # Reshape and combine data from all processes
            recvBuf = recvBuf.reshape(size, -1, shape[0])

            # Sum the collected arrays along the first axis
            uDense = np.sum(recvBuf, axis=0)
            uDense = [me.view(type(L.u[0])) for me in uDense]

        else:
            uDense = L.u if not L.sweep.coll.left_is_node else L.u[1:]

        self.add_to_stats(
            process=step.status.slot,
            time=L.time + L.dt,
            level=L.level_index,
            iter=step.status.iter,
            sweep=L.status.sweep,
            type='u_dense',
            value=uDense,
        )

        nodes = [L.time + L.dt * L.sweep.coll.nodes[m] for m in range(len(L.sweep.coll.nodes))]
        self.add_to_stats(
            process=step.status.slot,
            time=L.time + L.dt,
            level=L.level_index,
            iter=step.status.iter,
            sweep=L.status.sweep,
            type='nodes',
            value=np.append([L.time], nodes) if not L.sweep.coll.left_is_node else nodes,
        )


class LogSolutionAfterIteration(Hooks):
    """
    Store the solution at the end of each iteration as "u".
    """

    def post_iteration(self, step, level_number):
        """
        Record solution at the end of the iteration

        Parameters
        ----------
        step : pySDC.core.step.Step
            Current step
        level_number : int
            Current level number.

        Returns
        -------
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


class LogToFile(Hooks):
    r"""
    Hook for logging the solution to file after the step using pickle.

    Please configure the hook to your liking by manipulating class attributes.
    You must set a custom path to a directory like so:

    ```
    LogToFile.path = '/my/directory/'
    ```

    Keep in mind that the hook will overwrite files without warning!
    You can give a custom file name by setting the ``file_name`` class attribute and give a custom way of rendering the
    index associated with individual files by giving a different lambda function ``format_index`` class attribute. This
    lambda should accept one index and return one string.

    You can also give a custom ``logging_condition`` lambda, accepting the current level if you want to log selectively.

    Importantly, you may need to change ``process_solution``. By default, this will return a numpy view of the solution.
    Of course, if you are not using numpy, you need to change this. Again, this is a lambda accepting the level.

    After the fact, you can use the classmethod `get_path` to get the path to a certain data or the `load` function to
    directly load the solution at a given index. Just configure the hook like you did when you recorded the data
    beforehand.

    Finally, be aware that using this hook with MPI parallel runs may lead to different tasks overwriting files. Make
    sure to give a different `file_name` for each task that writes files.
    """

    path = None
    file_name = 'solution'
    logging_condition = lambda L: True
    process_solution = lambda L: {'t': L.time + L.dt, 'u': L.uend.view(np.ndarray)}
    format_index = lambda index: f'{index:06d}'

    def __init__(self):
        """Initialization routine"""
        super().__init__()
        self.counter = 0

        if self.path is None:
            raise ValueError('Please set a path for logging as the class attribute `LogToFile.path`!')

        if os.path.isfile(self.path):
            raise ValueError(
                f'{self.path!r} is not a valid path to log to because a file of the same name exists. Please supply a directory'
            )

        if not os.path.isdir(self.path):
            os.mkdir(self.path)

    def post_step(self, step, level_number):
        """
        Log solution at the end of the step in a file.

        Parameters
        ----------
        step : pySDC.core.step.Step
            Current step.
        level_number : int
            Current level number.

        Returns
        -------
        None
        """

        if level_number > 0:
            return None

        L = step.levels[level_number]

        if type(self).logging_condition(L):
            path = self.get_path(self.counter)
            data = type(self).process_solution(L)

            with open(path, 'wb') as file:
                pickle.dump(data, file)

            self.counter += 1

    @classmethod
    def get_path(cls, index):
        """
        Get path of the file where the solution is stored.

        Parameters
        ----------
        index : int
            Index of a file.

        Returns
        -------
        str : 
            Path of file.
        """
        return f'{cls.path}/{cls.file_name}_{cls.format_index(index)}.pickle'

    @classmethod
    def load(cls, index):
        """
        Load data from file.

        Parameters
        ----------
        index : int
            Index of a file.

        Returns
        -------
        file object :
            File where the solution is stored.
        """
        path = cls.get_path(index)
        with open(path, 'rb') as file:
            return pickle.load(file)
