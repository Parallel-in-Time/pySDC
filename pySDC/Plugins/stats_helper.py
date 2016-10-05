def filter_stats(stats,process=None,time=None,level=None,iter=None,type=None):
    """
    Helper function to extract data from the dictrionary of statistics

    Args:
        time: the requested simulation time
        level: the requested level id
        iter: the requested iteration count
        type: string to describe the requested type of value
    Returns:
        dictionary containing only the entries corresponding to the filter
    """

    result = {}
    for k,v in stats.items():
        # get data if key matches the filter (if specified)
        if (k.time == time or time is None) and \
                (k.process == process or process is None) and \
                (k.level == level or level is None) and \
                (k.iter == iter or iter is None) and \
                (k.type == type or type is None):
            result[k] = v

    return result


def sort_stats(stats,sortby):
    """
    Helper function to transform stats dictionary to sorted list of tuples

    Args:
        stats: dictionary of statistics
        sortby: string to specify which key to use for sorting
    """

    result = []
    for k,v in stats.items():
        # convert string to attribute and append key + value to result as tuple
        item = getattr(k,sortby)
        result.append((item,v))

    # sort by first element of the tuple (which is the sortby key) and return
    return sorted(result,key=lambda tup: tup[0])


def get_list_of_types(stats):
    """
    Helper function to get list of types registered in stats

    Args:
        stats: dictionary with statistics

    Returns:
        list of types registered

    """

    type_list = []
    for k, v in stats.items():
        if not k.type in type_list:
            type_list.append(k.type)
    return type_list
