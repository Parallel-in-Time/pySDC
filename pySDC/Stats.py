

class step_stats():
    """
    Statistics for a single time step

    Attributes:
        niter: number of iterations until convergence
        residual: residual after convergence
        level_stats: list of statistics per level
    """

    def __init__(self):
        """
        Initialization routine
        """
        self.niter = None
        self.residual = None
        self.level_stats = []

    def register_level_stats(self,les):
        """
        Registration of level statistics (will be a link to the levels)
        """
        assert isinstance(les,level_stats)
        self.level_stats.append(les)


class level_stats():
    """
    Statistics for a single level

    Attributes:
        residual: residual after convergence at this level
        iter_stats: list of statistics per iteration
    """

    def __init__(self):
        """
        Initialization routine
        """
        self.residual = None
        self.iter_stats = []

    def add_iter_stats(self):
        """
        Register iteration statistics
        """
        self.iter_stats.append(iter_stats())


class iter_stats():
    """
    Statistics for a single level

    Attributes:
        residual: residual after current iteration
    """

    def __init__(self):
        """
        Initialization routine
        """
        self.residual = None
