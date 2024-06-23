import numpy as np

from pySDC.core.errors import TransferError
from pySDC.core.space_transfer import SpaceTransfer


class mesh_to_mesh_fft(SpaceTransfer):
    """
    Custom base_transfer class, implements Transfer.py

    This implementation can restrict and prolong between 1d meshes with FFT for periodic boundaries

    Attributes:
        irfft_object_fine: planned FFT for backward transformation, real-valued output
        rfft_object_coarse: planned real-valued FFT for forward transformation
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
        super().__init__(fine_prob, coarse_prob, params)

        self.ratio = int(self.fine_prob.nvars / self.coarse_prob.nvars)

    def restrict(self, F):
        """
        Restriction implementation

        Args:
            F: the fine level data (easier to access than via the fine attribute)
        """
        G = type(F)(self.coarse_prob.init, val=0.0)

        if type(F).__name__ == 'mesh':
            G[:] = F[:: self.ratio]
        elif type(F).__name__ == 'imex_mesh':
            G.impl[:] = F.impl[:: self.ratio]
            G.expl[:] = F.expl[:: self.ratio]
        else:
            raise TransferError('Unknown data type, got %s' % type(F))
        return G

    def prolong(self, G):
        """
        Prolongation implementation

        Args:
            G: the coarse level data (easier to access than via the coarse attribute)
        """
        F = type(G)(self.fine_prob.init, val=0.0)

        def _prolong(coarse):
            coarse_hat = np.fft.rfft(coarse)
            fine_hat = np.zeros(self.fine_prob.init[0] // 2 + 1, dtype=np.complex128)
            half_idx = self.coarse_prob.init[0] // 2
            fine_hat[0:half_idx] = coarse_hat[0:half_idx]
            fine_hat[-1] = coarse_hat[-1]
            return np.fft.irfft(fine_hat) * self.ratio

        if type(G).__name__ == 'mesh':
            F[:] = _prolong(G)
        elif type(G).__name__ == 'imex_mesh':
            F.impl[:] = _prolong(G.impl)
            F.expl[:] = _prolong(G.expl)
        else:
            raise TransferError('Unknown data type, got %s' % type(G))
        return F
