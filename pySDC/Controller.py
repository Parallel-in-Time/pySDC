import abc
from future.utils import with_metaclass


class controller(with_metaclass(abc.ABCMeta)):
    """
    Base abstract controller class
    """

    def __init__(self):
        """
        Initialization routine for the base controller
        """
        pass


    def check_convergence(self, S):
        """
        Routine to determine whether to stop iterating (currently testing the residual and the max. number of iterations)

        Args:
            S: current step

        Returns:
            converged, true or false

        """

        # do all this on the finest level
        L = S.levels[0]

        # get residual and check against prescribed tolerance (plus check number of iterations
        res = L.status.residual
        converged = S.status.iter >= S.params.maxiter or res <= L.params.restol

        return converged

    @abc.abstractmethod
    def run(self, u0, t0, dt, Tend):
        """
        Abstract interface to the run() method

        Args:
            u0: initial values
            t0: starting time
            dt: (initial) time step
            Tend: ending time
        """
        return None
