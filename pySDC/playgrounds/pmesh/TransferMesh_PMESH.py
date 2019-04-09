from pySDC.core.Errors import TransferError
from pySDC.core.SpaceTransfer import space_transfer
from pySDC.playgrounds.pmesh.PMESH_datatype import pmesh_datatype, rhs_imex_pmesh


class pmesh_to_pmesh(space_transfer):
    """
    Custon base_transfer class, implements Transfer.py

    This implementation can restrict and prolong between 2d meshes with FFT for periodic boundaries

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
        super(pmesh_to_pmesh, self).__init__(fine_prob, coarse_prob, params)

    def restrict(self, F):
        """
        Restriction implementation

        Args:
            F: the fine level data (easier to access than via the fine attribute)
        """
        if isinstance(F, pmesh_datatype):
            G = self.coarse_prob.dtype_u(self.coarse_prob.init)
            F.values.resample(G.values)
            # G.values = self.coarse_prob.init.upsample(F.values, keep_mean=True)
        elif isinstance(F, rhs_imex_pmesh):
            G = self.coarse_prob.dtype_f(self.coarse_prob.init)
            F.impl.values.resample(G.impl.values)
            F.expl.values.resample(G.expl.values)
            # G.impl.values = self.coarse_prob.init.upsample(F.impl.values, keep_mean=True)
            # G.expl.values = self.coarse_prob.init.upsample(F.expl.values, keep_mean=True)
        else:
            raise TransferError('Unknown data type, got %s' % type(F))
        return G

    def prolong(self, G):
        """
        Prolongation implementation

        Args:
            G: the coarse level data (easier to access than via the coarse attribute)
        """
        if isinstance(G, pmesh_datatype):
            F = self.fine_prob.dtype_u(self.fine_prob.init)
            G.values.resample(F.values)
        elif isinstance(G, rhs_imex_pmesh):
            F = self.fine_prob.dtype_f(self.fine_prob.init)
            G.impl.values.resample(F.impl.values)
            G.expl.values.resample(F.expl.values)
        else:
            raise TransferError('Unknown data type, got %s' % type(G))
        return F
