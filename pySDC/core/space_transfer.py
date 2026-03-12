import logging
from typing import Any, Dict

from pySDC.helpers.pysdc_helper import FrozenClass


# short helper class to add params as attributes
class _Pars(FrozenClass):
    def __init__(self, pars: Dict[str, Any]) -> None:
        self.periodic: bool = False
        self.equidist_nested: bool = True
        self.iorder: int = 2
        self.rorder: int = 2
        for k, v in pars.items():
            setattr(self, k, v)
        # freeze class, no further attributes allowed from this point
        self._freeze()


class SpaceTransfer(object):
    """
    Abstract SpaceTransfer class

    Attributes:
        params (__Pars): parameters given by the user
        logger: custom logger for transfer-related logging
        fine_prob (pySDC.Problem.ptype): reference to the fine problem
        coarse_prob (pySDC.Problem.ptype): reference to the coarse problem
    """

    def __init__(self, fine_prob: Any, coarse_prob: Any, params: Dict[str, Any]) -> None:
        """
        Initialization routine

        Args:
            fine_prob (pySDC.Problem.ptype): reference to the fine problem
            coarse_prob (pySDC.Problem.ptype): reference to the coarse problem
            params (dict): user-defined parameters
        """

        self.params: _Pars = _Pars(params)

        # set up logger
        self.logger: logging.Logger = logging.getLogger('space-transfer')

        # just copy by object
        self.fine_prob: Any = fine_prob
        self.coarse_prob: Any = coarse_prob

    def restrict(self, F: Any) -> Any:
        """
        Abstract interface for restriction in space

        Args:
            F: the fine level data (easier to access than via the fine attribute)
        """
        raise NotImplementedError('ERROR: space_transfer has to implement restrict(self, F)')

    def prolong(self, G: Any) -> Any:
        """
        Abstract interface for prolongation in space

        Args:
            G: the coarse level data (easier to access than via the coarse attribute)
        """
        raise NotImplementedError('ERROR: space_transfer has to implement prolong(self, G)')
