

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

class stats_class():

    def __init__(self):
        self.__stats = {}

    def add_to_stats(self,time=-1,level=-1,iter=-1,key=-1,value=-1):

        self.__stats[(time,level,iter,key)] = value

    def return_stats(self):
        return self.__stats

stats = stats_class()