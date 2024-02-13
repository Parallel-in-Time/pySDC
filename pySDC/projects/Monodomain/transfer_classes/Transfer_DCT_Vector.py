import numpy as np
import scipy.fft as fft

from pySDC.core.Errors import TransferError
from pySDC.core.SpaceTransfer import space_transfer
from pySDC.projects.Monodomain.datatype_classes.FD_Vector import FD_Vector


class DCT_to_DCT(space_transfer):
    def __init__(self, fine_prob, coarse_prob, params):
        """
        Initialization routine

        Args:
            fine_prob: fine problem
            coarse_prob: coarse problem
            params: parameters for the transfer operators
        """

        # assumes problem has either periodic or neumann-zero boundary conditions

        # invoke super initialization
        super(DCT_to_DCT, self).__init__(fine_prob, coarse_prob, params)

        fine_grid = self.fine_prob.parabolic.grids
        coarse_grid = self.coarse_prob.parabolic.grids
        ndim = len(fine_grid)
        self.norm = "forward"

        assert ndim == 1, 'DCT transfer only implemented for 1D problems'

        # we have a 1d problem
        if ndim == 1:
            fine_grid = fine_grid[0]
            coarse_grid = coarse_grid[0]
            # if number of variables is the same on both levels, Rspace and Pspace are identity
            if fine_grid.size == coarse_grid.size:
                self.same_grid = True
            else:
                self.same_grid = False
        # we have an n-d problem
        else:
            raise TransferError('DCT transfer only implemented for 1D problems')

    def restrict(self, F):
        """
        Restriction implementation
        Args:
            F: the fine level data (easier to access than via the fine attribute)
        """

        if self.same_grid:
            G = F.copy()
        else:
            G = FD_Vector(self.coarse_prob.init)
            G.values[:] = fft.idct(fft.dct(F.values, norm=self.norm), n=G.values.size, norm=self.norm)

        return G

    def prolong(self, G):
        """
        Prolongation implementation
        Args:
            G: the coarse level data (easier to access than via the coarse attribute)
        """

        if self.same_grid:
            F = G.copy()
        else:
            F = FD_Vector(self.fine_prob.init)
            F.values[:] = fft.idct(fft.dct(G.values, norm=self.norm), n=F.values.size, norm=self.norm)

        return F
