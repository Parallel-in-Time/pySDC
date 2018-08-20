from __future__ import division

import numpy as np
import pyfftw


from pySDC.implementations.datatype_classes.mesh import mesh, rhs_imex_mesh
from pySDC.core.SpaceTransfer import space_transfer
from pySDC.core.Errors import TransferError


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
        assert len(self.fine_prob.params.nvars) == 2
        assert len(self.coarse_prob.params.nvars) == 2
        assert self.fine_prob.params.nvars[0] == self.fine_prob.params.nvars[1]
        assert self.coarse_prob.params.nvars[0] == self.coarse_prob.params.nvars[1]

        self.ratio = int(self.fine_prob.params.nvars[0] / self.coarse_prob.params.nvars[0])

        self.fft_in_fine = pyfftw.empty_aligned(self.fine_prob.init, dtype='complex128')
        self.fft_out_fine = pyfftw.empty_aligned(self.fine_prob.init, dtype='complex128')
        self.ifft_in_fine = pyfftw.empty_aligned(self.fine_prob.init, dtype='complex128')
        self.ifft_out_fine = pyfftw.empty_aligned(self.fine_prob.init, dtype='complex128')
        self.fft_object_fine = pyfftw.FFTW(self.fft_in_fine, self.fft_out_fine, direction='FFTW_FORWARD', axes=(0, 1))
        self.ifft_object_fine = pyfftw.FFTW(self.ifft_in_fine, self.ifft_out_fine, direction='FFTW_BACKWARD',
                                            axes=(0, 1))

        self.fft_in_coarse = pyfftw.empty_aligned(self.coarse_prob.init, dtype='complex128')
        self.fft_out_coarse = pyfftw.empty_aligned(self.coarse_prob.init, dtype='complex128')
        self.ifft_in_coarse = pyfftw.empty_aligned(self.coarse_prob.init, dtype='complex128')
        self.ifft_out_coarse = pyfftw.empty_aligned(self.coarse_prob.init, dtype='complex128')
        self.fft_object_coarse = pyfftw.FFTW(self.fft_in_coarse, self.fft_out_coarse, direction='FFTW_FORWARD',
                                             axes=(0, 1))
        self.ifft_object_coarse = pyfftw.FFTW(self.ifft_in_coarse, self.ifft_out_coarse, direction='FFTW_BACKWARD',
                                              axes=(0, 1))

    def restrict(self, F):
        """
        Restriction implementation

        Args:
            F: the fine level data (easier to access than via the fine attribute)
        """
        if isinstance(F, mesh):
            G = mesh(self.coarse_prob.init, val=0.0)
            G.values[:] = F.values[::self.ratio, ::self.ratio]
        elif isinstance(F, rhs_imex_mesh):
            G = rhs_imex_mesh(self.coarse_prob.init, val=0.0)
            G.impl.values = F.impl.values[::self.ratio, ::self.ratio]
            G.expl.values = F.expl.values[::self.ratio, ::self.ratio]
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
            tmpG = self.fft_object_coarse(G.values) / (self.coarse_prob.init[0] * self.coarse_prob.init[1])
            tmpF = np.zeros(self.fine_prob.init, dtype=np.complex128)
            halfG = int(self.coarse_prob.init[0] / 2)
            tmpF[0:halfG, 0:halfG] = tmpG[0:halfG, 0:halfG]
            tmpF[self.fine_prob.init[0] - halfG:, 0:halfG] = tmpG[halfG:, 0:halfG]
            tmpF[0:halfG, self.fine_prob.init[0] - halfG:] = tmpG[0:halfG, halfG:]
            tmpF[self.fine_prob.init[0] - halfG:, self.fine_prob.init[0] - halfG:] = tmpG[halfG:, halfG:]
            F.values[:] = np.real(self.ifft_object_fine(tmpF, normalise_idft=False))
        elif isinstance(G, rhs_imex_mesh):
            F = rhs_imex_mesh(G)
            tmpG_impl = self.fft_object_coarse(G.impl.values) / (self.coarse_prob.init[0] * self.coarse_prob.init[1])
            tmpF_impl = np.zeros(self.fine_prob.init, dtype=np.complex128)
            halfG = int(self.coarse_prob.init[0] / 2)
            tmpF_impl[0:halfG, 0:halfG] = tmpG_impl[0:halfG, 0:halfG]
            tmpF_impl[self.fine_prob.init[0] - halfG:, 0:halfG] = tmpG_impl[halfG:, 0:halfG]
            tmpF_impl[0:halfG, self.fine_prob.init[0] - halfG:] = tmpG_impl[0:halfG, halfG:]
            tmpF_impl[self.fine_prob.init[0] - halfG:, self.fine_prob.init[0] - halfG:] = tmpG_impl[halfG:, halfG:]
            F.impl.values[:] = np.real(self.ifft_object_fine(tmpF_impl, normalise_idft=False))
            tmpG_expl = self.fft_object_coarse(G.expl.values) / (self.coarse_prob.init[0] * self.coarse_prob.init[1])
            tmpF_expl = np.zeros(self.fine_prob.init, dtype=np.complex128)
            halfG = int(self.coarse_prob.init[0] / 2)
            tmpF_expl[0:halfG, 0:halfG] = tmpG_expl[0:halfG, 0:halfG]
            tmpF_expl[self.fine_prob.init[0] - halfG:, 0:halfG] = tmpG_expl[halfG:, 0:halfG]
            tmpF_expl[0:halfG, self.fine_prob.init[0] - halfG:] = tmpG_expl[0:halfG, halfG:]
            tmpF_expl[self.fine_prob.init[0] - halfG:, self.fine_prob.init[0] - halfG:] = tmpG_expl[halfG:, halfG:]
            F.expl.values[:] = np.real(self.ifft_object_fine(tmpF_expl, normalise_idft=False))
        else:
            raise TransferError('Unknown data type, got %s' % type(G))
        return F
