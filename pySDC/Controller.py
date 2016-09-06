import abc
from future.utils import with_metaclass


class controller(with_metaclass(abc.ABCMeta)):
    """
    Base abstract controller class

    Attributes:
        u0: initial values
        t0: starting time
        dt: (initial) time step
        Tend: ending time
    """

    def __init__(self, u0, t0, dt, Tend):
        """
        Initialization routine for the base controller

        Args:
            u0: initial values
            t0: starting time
            dt: (initial) time step
            Tend: ending time

        """

        self.u0 = u0
        self.t0 = t0
        self.dt = dt
        self.Tend = Tend


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
    def run(self):
        """
        Abstract interface to the run() method
        """
        return None


