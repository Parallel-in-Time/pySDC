

# class step_stats():
#     """
#     Statistics for a single time step
#
#     Attributes:
#         niter: number of iterations until convergence
#         residual: residual after convergence
#         level_stats: list of statistics per level
#     """
#
#     def __init__(self):
#         """
#         Initialization routine
#         """
#         self.niter = None
#         self.residual = None
#         self.level_stats = []
#
#     def add_level_stats(self):
#         """
#         Registration of level statistics
#         """
#         self.level_stats.append(level_stats())
#
#
# class level_stats():
#     """
#     Statistics for a single level
#
#     Attributes:
#         residual: residual after convergence at this level
#         iter_stats: list of statistics per iteration
#     """
#
#     def __init__(self):
#         """
#         Initialization routine
#         """
#         self.residual = None
#         self.iter_stats = []
#
#     def add_iter_stats(self):
#         """
#         Register iteration statistics
#         """
#         self.iter_stats.append(iter_stats())
#
#
# class iter_stats():
#     """
#     Statistics for a single level
#
#     Attributes:
#         residual: residual after current iteration
#     """
#
#     def __init__(self):
#         """
#         Initialization routine
#         """
#         self.residual = None


# sstats = []

from collections import namedtuple

class stats_class():

    def __init__(self):
        self.__stats = {}

    def add_to_stats(self,time=-1,level=-1,iter=-1,type=-1,value=-1):

        Entry = namedtuple('Entry',['time','level','iter','type'])
        self.__stats[Entry(time=time,level=level,iter=iter,type=type)] = value

    def return_stats(self):
        return self.__stats


def grep_stats(stats,time=None,level=None,iter=None,type=None):

    result = {}
    for k,v in stats.items():
        if (k.time == time or time is None) and \
                (k.level == level or level is None) and \
                (k.iter == iter or iter is None) and \
                (k.type == type or type is None):
            result[k] = v

    return result


def sort_stats(stats,sortby='time'):

    result = []

    for k,v in stats.items():
        result.append((getattr(k,sortby),v))

    return sorted(result,key=lambda tup: tup[0])




stats = stats_class()