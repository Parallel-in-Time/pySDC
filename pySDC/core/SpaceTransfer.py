import logging

from pySDC.helpers.pysdc_helper import FrozenClass


# short helper class to add params as attributes
class _Pars(FrozenClass):
    def __init__(self, pars):
        self.periodic = False
        self.equidist_nested = True
        self.iorder = 2
        self.rorder = 2
        for k, v in pars.items():
            setattr(self, k, v)
        # freeze class, no further attributes allowed from this point
        self._freeze()


class space_transfer(object):
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

        self.params = _Pars(space_transfer_params)

        # set up logger
        self.logger = logging.getLogger('space-transfer')

        # just copy by object
        self.fine_prob = fine_prob
        self.coarse_prob = coarse_prob

    def restrict(self, F):
        """
        Abstract interface for restriction in space

        Args:
            F: the fine level data (easier to access than via the fine attribute)
        """
        raise NotImplementedError('ERROR: space_transfer has to implement restrict(self, F)')

    def prolong(self, G):
        """
        Abstract interface for prolongation in space

        Args:
            G: the coarse level data (easier to access than via the coarse attribute)
        """
        raise NotImplementedError('ERROR: space_transfer has to implement prolong(self, G)')
