from pySDC.core.SpaceTransfer import space_transfer
from pySDC.projects.Monodomain.datatype_classes.myfloat import myfloat


class Transfer_myfloat(space_transfer):
    """
    This implementation can restrict and prolong between super vectors
    """

    def __init__(self, fine_prob, coarse_prob, params):
        """
        Initialization routine

        Args:
            fine_prob: fine problem
            coarse_prob: coarse problem
            params: parameters for the transfer operators
        """

        # invoke super initialization
        super(Transfer_myfloat, self).__init__(fine_prob, coarse_prob, params)

    def restrict(self, F):
        """
        Restriction implementation

        Args:
            F: the fine level data
        """
        u_coarse = myfloat(F)

        return u_coarse

    def prolong(self, G):
        """
        Prolongation implementation

        Args:
            G: the coarse level data
        """
        u_fine = myfloat(G)

        return u_fine
