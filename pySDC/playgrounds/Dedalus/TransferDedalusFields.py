
import numpy as np

from pySDC.core.Errors import TransferError
from pySDC.core.SpaceTransfer import space_transfer

from pySDC.playgrounds.Dedalus.dedalus_field import dedalus_field, rhs_imex_dedalus_field


class dedalus_field_transfer(space_transfer):
    """
    Custon base_transfer class, implements Transfer.py

    This implementation can restrict and prolong between 1d meshes with FFT for periodic boundaries

    Attributes:
        ratio: ratio between fine and coarse level
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
        super(dedalus_field_transfer, self).__init__(fine_prob, coarse_prob, params)

        self.ratio = list(fine_prob.init.global_grid_shape() / coarse_prob.init.global_grid_shape())

        assert self.ratio.count(self.ratio[0]) == len(self.ratio)

    def restrict(self, F):
        """
        Restriction implementation

        Args:
            F: the fine level data (easier to access than via the fine attribute)
        """
        if isinstance(F, dedalus_field):
            G = self.coarse_prob.dtype_u(self.coarse_prob.init)
            FG = self.fine_prob.init.new_field()
            FG['g'] = F.values['g']
            FG.set_scales(scales=1.0 / self.ratio[0])
            G.values['g'] = FG['g']
        elif isinstance(F, rhs_imex_dedalus_field):
            G = self.coarse_prob.dtype_f(self.coarse_prob.init)
            FG = self.fine_prob.init.new_field()
            FG['g'] = F.impl.values['g']
            FG.set_scales(scales=1.0 / self.ratio[0])
            G.impl.values['g'] = FG['g']
            FG = self.fine_prob.init.new_field()
            FG['g'] = F.expl.values['g']
            FG.set_scales(scales=1.0 / self.ratio[0])
            G.expl.values['g'] = FG['g']
        else:
            raise TransferError('Unknown data type, got %s' % type(F))
        return G

    def prolong(self, G):
        """
        Prolongation implementation

        Args:
            G: the coarse level data (easier to access than via the coarse attribute)
        """
        if isinstance(G, dedalus_field):
            F = self.fine_prob.dtype_u(self.fine_prob.init)
            GF = self.coarse_prob.init.new_field()
            GF['g'] = G.values['g']
            GF.set_scales(scales=self.ratio[0])
            F.values['g'] = GF['g']
        elif isinstance(G, rhs_imex_dedalus_field):
            F = self.fine_prob.dtype_f(self.fine_prob.init)
            GF = self.coarse_prob.init.new_field()
            GF['g'] = G.impl.values['g']
            GF.set_scales(scales=self.ratio[0])
            F.impl.values['g'] = GF['g']
            GF = self.coarse_prob.init.new_field()
            GF['g'] = G.expl.values['g']
            GF.set_scales(scales=self.ratio[0])
            F.expl.values['g'] = GF['g']
        else:
            raise TransferError('Unknown data type, got %s' % type(G))
        return F
