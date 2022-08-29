import numpy as np


def filter_stats(stats, process=None, time=None, level=None, iter=None, type=None, recomputed=None):
    """
    Helper function to extract data from the dictrionary of statistics

    Args:
        stats (dict): raw statistics from a controller run
        process (int): process number
        time (float): the requested simulation time
        level (int): the requested level index
        iter (int): the requested iteration count
        type (str): string to describe the requested type of value
        recomputed (bool): filter out intermediate values that have no impact on the solution because the associated
                           step was restarted if True. (Or filter the restarted if False. Use None to get both.)
    Returns:
        dict: dictionary containing only the entries corresponding to the filter
    """
    result = {}

    # check which steps have been recomputed
    if recomputed is not None:
        # this will contain a 2d array with all times and whether they have been recomputed
        restarts = np.array(get_sorted(stats, process=None, time=None, iter=None, type='recomputed',
                            recomputed=None, sortby='time'))
    else:
        # dummy values for when no filtering of restarts is desired
        restarts = np.array([[None, None]])

    for k, v in stats.items():
        # get data if key matches the filter (if specified)
        if (k.time == time or time is None) and \
                (k.process == process or process is None) and \
                (k.level == level or level is None) and \
                (k.iter == iter or iter is None) and \
                (k.type == type or type is None):

            if k.time in restarts[:, 0]:
                # we know there is only one entry for each time, so we make a mask for the time and take the first and
                # only entry and then take the second entry of this, which contains whether a restart was performed at
                # this time as a float and compare it to the value we specified for recomputed
                if restarts[restarts[:, 0] == k.time][0][1] == float(recomputed):
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


def get_sorted(stats, process=None, time=None, level=None, iter=None, type=None, recomputed=None, sortby='time'):
    return sort_stats(filter_stats(stats, process=process, time=time, level=level, iter=iter, type=type,
                      recomputed=recomputed), sortby=sortby)
