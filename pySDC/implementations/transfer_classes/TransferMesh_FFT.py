
import numpy as np

from pySDC.core.Errors import TransferError
from pySDC.core.SpaceTransfer import space_transfer
from pySDC.implementations.datatype_classes.mesh import mesh, imex_mesh


class mesh_to_mesh_fft(space_transfer):
    """
    Custon base_transfer class, implements Transfer.py

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
        super(mesh_to_mesh_fft, self).__init__(fine_prob, coarse_prob, params)

        self.ratio = int(self.fine_prob.params.nvars / self.coarse_prob.params.nvars)

    def restrict(self, F):
        """
        Restriction implementation

        Args:
            F: the fine level data (easier to access than via the fine attribute)
        """
        if isinstance(F, mesh):
            G = mesh(self.coarse_prob.init, val=0.0)
            G[:] = F[::self.ratio]
        elif isinstance(F, imex_mesh):
            G = imex_mesh(self.coarse_prob.init, val=0.0)
            G.impl[:] = F.impl[::self.ratio]
            G.expl[:] = F.expl[::self.ratio]
        else:
            raise TransferError('Unknown data type, got %s' % type(F))
        return G

    def prolong(self, G):
        """
        Prolongation implementation

        Args:
            G: the coarse level data (easier to access than via the coarse attribute)
        """
        if isinstance(G, mesh):
            F = mesh(self.fine_prob.init, val=0.0)
            tmpG = np.fft.rfft(G)
            tmpF = np.zeros(self.fine_prob.init[0] // 2 + 1, dtype=np.complex128)
            halfG = int(self.coarse_prob.init[0] / 2)
            tmpF[0: halfG] = tmpG[0: halfG]
            tmpF[-1] = tmpG[-1]
            F[:] = np.fft.irfft(tmpF) * self.ratio
        elif isinstance(G, imex_mesh):
            F = imex_mesh(G)
            tmpG_impl = np.fft.rfft(G.impl)
            tmpF_impl = np.zeros(self.fine_prob.init[0] // 2 + 1, dtype=np.complex128)
            halfG = int(self.coarse_prob.init[0] / 2)
            tmpF_impl[0: halfG] = tmpG_impl[0: halfG]
            tmpF_impl[-1] = tmpG_impl[-1]
            F.impl[:] = np.fft.irfft(tmpF_impl) * self.ratio
            tmpG_expl = np.fft.rfft(G.expl)
            tmpF_expl = np.zeros(self.fine_prob.init[0] // 2 + 1, dtype=np.complex128)
            halfG = int(self.coarse_prob.init[0] / 2)
            tmpF_expl[0: halfG] = tmpG_expl[0: halfG]
            tmpF_expl[-1] = tmpG_expl[-1]
            F.expl[:] = np.fft.irfft(tmpF_expl) * self.ratio
        else:
            raise TransferError('Unknown data type, got %s' % type(G))
        return F
