import abc
from future.utils import with_metaclass
import logging

from pySDC.plugins.pysdc_helper import FrozenClass


class space_transfer(with_metaclass(abc.ABCMeta)):
    """
    Abstract space_transfer class

    Attributes:
        params (__Pars): parameters given by the user
        logger: custom logger for transfer-related logging
        fine_prob (pySDC.Problem.ptype): reference to the fine problem
        coarse_prob (pySDC.Problem.ptype): reference to the coarse problem
    """

    def __init__(self, fine_prob, coarse_prob, space_transfer_params):
        """
        Initialization routine

        Args:
            fine_prob (pySDC.Problem.ptype): reference to the fine problem
            coarse_prob (pySDC.Problem.ptype): reference to the coarse problem
            space_transfer_params (dict): user-defined parameters
        """

        # short helper class to add params as attributes
        class __Pars(FrozenClass):
            def __init__(self, pars):
                self.finter = False
                self.periodic = False
                for k, v in pars.items():
                    setattr(self, k, v)
                # freeze class, no further attributes allowed from this point
                self._freeze()

        self.params = __Pars(space_transfer_params)

        # set up logger
        self.logger = logging.getLogger('space-transfer')

        # just copy by object
        self.fine_prob = fine_prob
        self.coarse_prob = coarse_prob

    @abc.abstractmethod
    def restrict(self, F):
        """
        Abstract interface for restriction in space

        Args:
            F: the fine level data (easier to access than via the fine attribute)
        """
        pass

    @abc.abstractmethod
    def prolong(self, G):
        """
        Abstract interface for prolongation in space

        Args:
            G: the coarse level data (easier to access than via the coarse attribute)
        """
        pass
