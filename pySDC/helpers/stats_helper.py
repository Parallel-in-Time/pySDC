import numpy as np


def filter_stats(stats, comm=None, recomputed=None, **kwargs):
    """
    Helper function to extract data from the dictionary of statistics. Please supply metadata as keyword arguments.

    Args:
        stats (dict): raw statistics from a controller run
        recomputed (bool): filter recomputed values from stats if set to anything other than None
        comm (mpi4py.MPI.Intracomm): Communicator (or None if not applicable)

    Returns:
        dict: dictionary containing only the entries corresponding to the filter
    """
    result = {}
    for k, v in stats.items():
        if all([k._asdict().get(k2, None) == v2 for k2, v2 in kwargs.items() if v2 is not None] + [True]):
            result[k] = v

    if comm is not None:
        # gather the results across all ranks
        result = {key: value for sub_result in comm.allgather(result) for key, value in sub_result.items()}

    if recomputed is not None:
        # delete values that have been recorded and superseded by similar, but not identical keys
        times_restarted = np.unique([me.time for me in result.keys() if me.num_restarts > 0])
        for t in times_restarted:
            restarts = {}
            for me in filter_stats(result, time=t).keys():
                restarts[me.type] = max([restarts.get(me.type, 0), me.num_restarts])

            [
                [
                    [result.pop(you, None) for you in filter_stats(result, time=t, type=type_, num_restarts=i).keys()]
                    for i in range(num_restarts_)
                ]
                for type_, num_restarts_ in restarts.items()
            ]

        # delete values that were recorded at times that shouldn't be recorded because we performed a different step after the restart
        if kwargs.get('type', None) != '_recomputed':
            other_restarted_steps = [
                key for key, val in filter_stats(stats, type='_recomputed', recomputed=False, comm=comm).items() if val
            ]
            for step in other_restarted_steps:
                [result.pop(me) for me in filter_stats(result, time=step.time).keys()]

    return result


def sort_stats(stats, sortby):
    """
    Helper function to transform stats dictionary to sorted list of tuples

    Args:
        stats (dict): dictionary of statistics
        sortby (str): string to specify which key to use for sorting

    Returns:
        list: list of tuples containing the sortby item and the value
    """

    result = []
    for k, v in stats.items():
        # convert string to attribute and append key + value to result as tuple
        item = getattr(k, sortby)
        result.append((item, v))

    # sort by first element of the tuple (which is the sortby key) and return
    sorted_data = sorted(result, key=lambda tup: tup[0])

    return sorted_data


def get_list_of_types(stats):
    """
    Helper function to get list of types registered in stats

    Args:
        stats (dict): dictionary with statistics

    Returns:
        list: list of types registered

    """

    type_list = []
    for k, _ in stats.items():
        if k.type not in type_list:
            type_list.append(k.type)
    return type_list


def get_sorted(stats, sortby='time', **kwargs):
    """
    Utility for filtering and sorting stats in a single call. Pass a communicator if using MPI.
    Keyword arguments are passed to `filter_stats` for filtering.

    stats (dict): raw statistics from a controller run
    sortby (str): string to specify which key to use for sorting

    Returns:
        list: list of tuples containing the sortby item and the value
    """

    return sort_stats(
        filter_stats(stats, **kwargs),
        sortby=sortby,
    )
