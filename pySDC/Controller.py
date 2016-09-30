import abc
from future.utils import with_metaclass

from pySDC.Plugins.pysdc_helper import FrozenClass


class controller(with_metaclass(abc.ABCMeta)):
    """
    Base abstract controller class
    """

    def __init__(self, controller_params):
        """
        Initialization routine for the base controller

        Args:
            controller_params: parameter set for the controller and the steps
        """

        # short helper class to add params as attributes
        class pars(FrozenClass):
            def __init__(self, params):

                self.fine_comm = True
                self.predict = True

                for k, v in params.items():
                    setattr(self, k, v)

                self._freeze()

        self.params = pars(controller_params)
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
    def run(self, u0, t0, Tend):
        """
        Abstract interface to the run() method

        Args:
            u0: initial values
            t0: starting time
            Tend: ending time
        """
        return None
