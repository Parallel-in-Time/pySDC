import numpy as np

from pySDC.core.Errors import TransferError
from pySDC.core.SpaceTransfer import space_transfer
from pySDC.implementations.datatype_classes.mesh import mesh, imex_mesh


class mesh_to_mesh_fft2d(space_transfer):
    """
    Custon base_transfer class, implements Transfer.py

    This implementation can restrict and prolong between 2d meshes with FFT for periodic boundaries

    Attributes:
        Rspace: spatial restriction matrix, dim. Nf x Nc
        Pspace: spatial prolongation matrix, dim. Nc x Nf
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
        super(mesh_to_mesh_fft2d, self).__init__(fine_prob, coarse_prob, params)

        # TODO: cleanup and move to real-valued FFT
        assert len(self.fine_prob.nvars) == 2
        assert len(self.coarse_prob.nvars) == 2
        assert self.fine_prob.nvars[0] == self.fine_prob.nvars[1]
        assert self.coarse_prob.nvars[0] == self.coarse_prob.nvars[1]

        self.ratio = int(self.fine_prob.nvars[0] / self.coarse_prob.nvars[0])

    def restrict(self, F):
        """
        Restriction implementation

        Args:
            F: the fine level data (easier to access than via the fine attribute)
        """
        if isinstance(F, mesh):
            G = mesh(self.coarse_prob.init, val=0.0)
            G[:] = F[:: self.ratio, :: self.ratio]
        elif isinstance(F, imex_mesh):
            G = imex_mesh(self.coarse_prob.init, val=0.0)
            G.impl[:] = F.impl[:: self.ratio, :: self.ratio]
            G.expl[:] = F.expl[:: self.ratio, :: self.ratio]
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
            F = mesh(self.fine_prob.init)
            tmpG = np.fft.fft2(G)
            tmpF = np.zeros(self.fine_prob.init[0], dtype=np.complex128)
            halfG = int(self.coarse_prob.init[0][0] / 2)
            tmpF[0:halfG, 0:halfG] = tmpG[0:halfG, 0:halfG]
            tmpF[self.fine_prob.init[0][0] - halfG :, 0:halfG] = tmpG[halfG:, 0:halfG]
            tmpF[0:halfG, self.fine_prob.init[0][0] - halfG :] = tmpG[0:halfG, halfG:]
            tmpF[self.fine_prob.init[0][0] - halfG :, self.fine_prob.init[0][0] - halfG :] = tmpG[halfG:, halfG:]
            F[:] = np.real(np.fft.ifft2(tmpF)) * self.ratio * 2
        elif isinstance(G, imex_mesh):
            F = imex_mesh(G)
            tmpG_impl = np.fft.fft2(G.impl)
            tmpF_impl = np.zeros(self.fine_prob.init, dtype=np.complex128)
            halfG = int(self.coarse_prob.init[0][0] / 2)
            tmpF_impl[0:halfG, 0:halfG] = tmpG_impl[0:halfG, 0:halfG]
            tmpF_impl[self.fine_prob.init[0][0] - halfG :, 0:halfG] = tmpG_impl[halfG:, 0:halfG]
            tmpF_impl[0:halfG, self.fine_prob.init[0][0] - halfG :] = tmpG_impl[0:halfG, halfG:]
            tmpF_impl[self.fine_prob.init[0][0] - halfG :, self.fine_prob.init[0][0] - halfG :] = tmpG_impl[
                halfG:, halfG:
            ]
            F.impl[:] = np.real(np.fft.ifft2(tmpF_impl)) * self.ratio * 2
            tmpG_expl = np.fft.fft2(G.expl) / (self.coarse_prob.init[0] * self.coarse_prob.init[1])
            tmpF_expl = np.zeros(self.fine_prob.init[0], dtype=np.complex128)
            halfG = int(self.coarse_prob.init[0][0] / 2)
            tmpF_expl[0:halfG, 0:halfG] = tmpG_expl[0:halfG, 0:halfG]
            tmpF_expl[self.fine_prob.init[0][0] - halfG :, 0:halfG] = tmpG_expl[halfG:, 0:halfG]
            tmpF_expl[0:halfG, self.fine_prob.init[0][0] - halfG :] = tmpG_expl[0:halfG, halfG:]
            tmpF_expl[self.fine_prob.init[0][0] - halfG :, self.fine_prob.init[0][0] - halfG :] = tmpG_expl[
                halfG:, halfG:
            ]
            F.expl[:] = np.real(np.fft.ifft2(tmpF_expl)) * self.ratio * 2
        else:
            raise TransferError('Unknown data type, got %s' % type(G))
        return F
