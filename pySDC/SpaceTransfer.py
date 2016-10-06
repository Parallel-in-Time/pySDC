import abc
from future.utils import with_metaclass

from pySDC.Plugins.pysdc_helper import FrozenClass


class space_transfer(with_metaclass(abc.ABCMeta)):
    """
    Abstract space_transfer class

    Attributes:
        fine: reference to the fine level
        coarse: reference to the coarse level
        init_f: number of variables on the fine level (whatever init represents there)
        init_c: number of variables on the coarse level (whatever init represents there)
    """

    def __init__(self,fine_prob,coarse_prob,space_transfer_params):
        """
        Initialization routine

        Args:

        """

        # short helper class to add params as attributes
        class pars(FrozenClass):
            def __init__(self,params):

                self.finter = False

                for k,v in params.items():
                    setattr(self,k,v)

                self._freeze()

        self.params = pars(space_transfer_params)

        # just copy by object
        self.fine_prob = fine_prob
        self.coarse_prob = coarse_prob


    @abc.abstractmethod
    def restrict(self,F):
        """
        Abstract interface for restriction in space

        Args:
            F: the fine level data (easier to access than via the fine attribute)
        """
        pass

    @abc.abstractmethod
    def prolong(self,G):
        """
        Abstract interface for prolongation in space

        Args:
            G: the coarse level data (easier to access than via the coarse attribute)
        """
        pass



