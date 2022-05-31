import numpy as np


def filter_stats(stats, process=None, time=None, level=None, iter=None, type=None, recomputed=False):
    """
    Helper function to extract data from the dictrionary of statistics

    Args:
        stats (dict): raw statistics from a controller run
        process (int): process number
        time (float): the requested simulation time
        level (int): the requested level index
        iter (int): the requested iteration count
        type (str): string to describe the requested type of value
    Returns:
        dict: dictionary containing only the entries corresponding to the filter
    """
    result = {}

    # check which steps have been recomputed
    if recomputed is not None:
        restarts = np.array(sort_stats(filter_stats(stats, process=None, time=None, iter=None, type='recomputed',
                            recomputed=None), sortby='time'))
    else:
        restarts = np.array([[None, None]])

    for k, v in stats.items():
        # get data if key matches the filter (if specified)
        if (k.time == time or time is None) and \
                (k.process == process or process is None) and \
                (k.level == level or level is None) and \
                (k.iter == iter or iter is None) and \
                (k.type == type or type is None):

            if k.time in restarts[:, 0]:
                if restarts[restarts[:, 0] == k.time][0, 1] == 0:
                    result[k] = v
            else:
                result[k] = v

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
    return sorted(result, key=lambda tup: tup[0])


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
