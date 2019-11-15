
import numpy as np

from pySDC.core.Errors import TransferError
from pySDC.core.SpaceTransfer import space_transfer

from pySDC.implementations.datatype_classes.parallel_mesh import parallel_mesh, parallel_imex_mesh


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

        self.ratio = list(fine_prob.domain.global_grid_shape() / coarse_prob.domain.global_grid_shape())

        assert self.ratio.count(self.ratio[0]) == len(self.ratio)

    def restrict(self, F):
        """
        Restriction implementation

        Args:
            F: the fine level data (easier to access than via the fine attribute)
        """
        if isinstance(F, parallel_mesh):
            G = self.coarse_prob.dtype_u(self.coarse_prob.init)
            for l in range(self.coarse_prob.init[0][-1]):
                FG = self.fine_prob.domain.new_field()
                FG['g'] = F[..., l]
                FG.set_scales(scales=1.0 / self.ratio[0])
                G[..., l] = FG['g']
        elif isinstance(F, parallel_imex_mesh):
            G = self.coarse_prob.dtype_f(self.coarse_prob.init)
            for l in range(self.fine_prob.init[0][-1]):
                FG = self.fine_prob.domain.new_field()
                FG['g'] = F.impl[..., l]
                FG.set_scales(scales=1.0 / self.ratio[0])
                G.impl[..., l] = FG['g']
                FG = self.fine_prob.domain.new_field()
                FG['g'] = F.expl[..., l]
                FG.set_scales(scales=1.0 / self.ratio[0])
                G.expl[..., l] = FG['g']
        else:
            raise TransferError('Unknown data type, got %s' % type(F))
        return G

    def prolong(self, G):
        """
        Prolongation implementation

        Args:
            G: the coarse level data (easier to access than via the coarse attribute)
        """
        if isinstance(G, parallel_mesh):
            F = self.fine_prob.dtype_u(self.fine_prob.init)
            for l in range(self.fine_prob.init[0][-1]):
                GF = self.coarse_prob.domain.new_field()
                GF['g'] = G[..., l]
                GF.set_scales(scales=self.ratio[0])
                F[..., l] = GF['g']
        elif isinstance(G, parallel_imex_mesh):
            F = self.fine_prob.dtype_f(self.fine_prob.init)
            for l in range(self.coarse_prob.init[0][-1]):
                GF = self.coarse_prob.domain.new_field()
                GF['g'] = G.impl[..., l]
                GF.set_scales(scales=self.ratio[0])
                F.impl[..., l] = GF['g']
                GF = self.coarse_prob.init[0].new_field()
                GF['g'] = G.expl[..., l]
                GF.set_scales(scales=self.ratio[0])
                F.expl[..., l] = GF['g']
        else:
            raise TransferError('Unknown data type, got %s' % type(G))
        return F
