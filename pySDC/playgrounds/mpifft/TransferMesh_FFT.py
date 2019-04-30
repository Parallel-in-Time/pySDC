from pySDC.core.Errors import TransferError
from pySDC.core.SpaceTransfer import space_transfer
from pySDC.playgrounds.mpifft.FFT_datatype import fft_datatype, rhs_imex_fft
from mpi4py_fft import PFFT
import numpy as np
import time


class fft_to_fft(space_transfer):
    """
    Custon base_transfer class, implements Transfer.py

    This implementation can restrict and prolong between PMESH datatypes meshes with FFT for periodic boundaries

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
        super(fft_to_fft, self).__init__(fine_prob, coarse_prob, params)

        assert self.fine_prob.params.spectral == self.coarse_prob.params.spectral

        self.spectral = self.fine_prob.params.spectral

        Nf = list(self.fine_prob.fft.global_shape())
        Nc = list(self.coarse_prob.fft.global_shape())
        self.ratio = [int(nf / nc) for nf, nc in zip(Nf, Nc)]
        axes = tuple(range(len(Nf)))
        self.fft_pad = PFFT(self.coarse_prob.params.comm, Nc, padding=self.ratio, axes=axes, dtype=np.float, slab=True)

    def restrict(self, F):
        """
        Restriction implementation

        Args:
            F: the fine level data (easier to access than via the fine attribute)
        """
        if isinstance(F, fft_datatype):
            if self.spectral:
                G = self.coarse_prob.dtype_u(self.coarse_prob.init)
                tmpF = self.fine_prob.fft.backward(F)
                tmpG = tmpF[::int(self.ratio[0]), ::int(self.ratio[1])]
                G[:] = self.coarse_prob.fft.forward(tmpG, G)
            else:
                G = self.coarse_prob.dtype_u(self.coarse_prob.init)
                G[:] = F[::int(self.ratio[0]), ::int(self.ratio[1])]
        else:
            raise TransferError('Unknown data type, got %s' % type(F))

        return G

    def prolong(self, G):
        """
        Prolongation implementation

        Args:
            G: the coarse level data (easier to access than via the coarse attribute)
        """
        if isinstance(G, fft_datatype):
            if self.spectral:
                F = self.fine_prob.dtype_u(self.fine_prob.init)
                tmpF = self.fft_pad.backward(G)
                F[:] = self.fine_prob.fft.forward(tmpF, F)
            else:
                F = self.fine_prob.dtype_u(self.fine_prob.init)
                G_hat = self.coarse_prob.fft.forward(G)
                F[:] = self.fft_pad.backward(G_hat, F)
        elif isinstance(G, rhs_imex_fft):
            if self.spectral:
                F = self.fine_prob.dtype_f(self.fine_prob.init)
                tmpF = self.fft_pad.backward(G.impl)
                F.impl[:] = self.fine_prob.fft.forward(tmpF, F.impl)
                tmpF = self.fft_pad.backward(G.expl)
                F.expl[:] = self.fine_prob.fft.forward(tmpF, F.expl)
            else:
                F = self.fine_prob.dtype_f(self.fine_prob.init)
                G_hat = self.coarse_prob.fft.forward(G.impl)
                F.impl[:] = self.fft_pad.backward(G_hat, F.impl)
                G_hat = self.coarse_prob.fft.forward(G.expl)
                F.expl[:] = self.fft_pad.backward(G_hat, F.expl)
        else:
            raise TransferError('Unknown data type, got %s' % type(G))

        return F
