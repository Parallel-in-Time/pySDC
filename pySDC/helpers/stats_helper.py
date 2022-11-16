import numpy as np


def filter_stats(stats, process=None, time=None, level=None, iter=None, type=None, recomputed=None, num_restarts=None):
    """
    Helper function to extract data from the dictrionary of statistics

    Args:
        stats (dict): raw statistics from a controller run
        process (int): process number
        time (float): the requested simulation time
        level (int): the requested level index
        iter (int): the requested iteration count
        type (str): string to describe the requested type of value
        recomputed (bool): filter recomputed values from stats if set to anything other than None

    Returns:
        dict: dictionary containing only the entries corresponding to the filter
    """
    result = {}

    for k, v in stats.items() if recomputed is None else filter_recomputed(stats.copy()).items():
        # get data if key matches the filter (if specified)
        if (
            (k.time == time or time is None)
            and (k.process == process or process is None)
            and (k.level == level or level is None)
            and (k.iter == iter or iter is None)
            and (k.type == type or type is None)
            and (k.num_restarts == num_restarts or num_restarts is None)
        ):
            result[k] = v

    return result


def sort_stats(stats, sortby, comm=None):
    """
    Helper function to transform stats dictionary to sorted list of tuples

    Args:
        stats (dict): dictionary of statistics
        sortby (str): string to specify which key to use for sorting
        comm (mpi4py.MPI.Intracomm): Communicator (or None if not applicable)

    Returns:
        list: list of tuples containing the sortby item and the value
    """

    result = []
    for k, v in stats.items():
        # convert string to attribute and append key + value to result as tuple
        item = getattr(k, sortby)
        result.append((item, v))

    if comm is not None:
        # gather the results across all ranks and the flatten the list
        result = [item for sub_result in comm.allgather(result) for item in sub_result]

    # sort by first element of the tuple (which is the sortby key) and return
    sorted_data = sorted(result, key=lambda tup: tup[0])

    return sorted_data


def filter_recomputed(stats):
    """
    Filter recomputed values from the stats and remove them.

    Args:
        stats (dict): Raw statistics from a controller run

    Returns:
        dict: The filtered stats dict
    """

    # delete values that have been recorded and superseded by similar, but not identical keys
    times_restarted = [me.time for me in stats.keys() if me.num_restarts > 0]
    for t in times_restarted:
        restarts = max([me.num_restarts for me in filter_stats(stats, type='_recomputed', time=t).keys()])
        for i in range(restarts):
            [stats.pop(me) for me in filter_stats(stats, time=t, num_restarts=i).keys()]

    # delete values that were recorded at times that shouln't be recorded because we performed a different step after the restart
    other_restarted_steps = [me for me in filter_stats(stats, type='_recomputed') if stats[me]]
    for step in other_restarted_steps:
        [stats.pop(me) for me in filter_stats(stats, time=step.time).keys()]

    return stats


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


def get_sorted(stats, sortby='time', comm=None, **kwargs):
    """
    Utility for filtering and sorting stats in a single call. Pass a communicatior if using MPI.
    Keyword arguments are passed to `filter_stats` for filtering.

    stats (dict): raw statistics from a controller run
    sortby (str): string to specify which key to use for sorting
    comm (mpi4py.MPI.Intracomm): Communicator (or None if not applicable)

    Returns:
        list: list of tuples containing the sortby item and the value
    """

    return sort_stats(
        filter_stats(stats, **kwargs),
        sortby=sortby,
        comm=comm,
    )
