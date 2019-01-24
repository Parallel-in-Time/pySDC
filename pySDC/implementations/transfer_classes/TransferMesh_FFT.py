
import numpy as np
import pyfftw

from pySDC.core.Errors import TransferError
from pySDC.core.SpaceTransfer import space_transfer
from pySDC.implementations.datatype_classes.mesh import mesh, rhs_imex_mesh


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

        ifft_in_fine = pyfftw.empty_aligned(self.fine_prob.init // 2 + 1, dtype='complex128')
        irfft_out_fine = pyfftw.empty_aligned(self.fine_prob.init, dtype='float64')
        self.irfft_object_fine = pyfftw.FFTW(ifft_in_fine, irfft_out_fine, direction='FFTW_BACKWARD')

        rfft_in_coarse = pyfftw.empty_aligned(self.coarse_prob.init, dtype='float64')
        fft_out_coarse = pyfftw.empty_aligned(self.coarse_prob.init // 2 + 1, dtype='complex128')
        self.rfft_object_coarse = pyfftw.FFTW(rfft_in_coarse, fft_out_coarse, direction='FFTW_FORWARD')

    def restrict(self, F):
        """
        Restriction implementation

        Args:
            F: the fine level data (easier to access than via the fine attribute)
        """
        if isinstance(F, mesh):
            G = mesh(self.coarse_prob.init, val=0.0)
            G.values = F.values[::self.ratio]
        elif isinstance(F, rhs_imex_mesh):
            G = rhs_imex_mesh(self.coarse_prob.init, val=0.0)
            G.impl.values = F.impl.values[::self.ratio]
            G.expl.values = F.expl.values[::self.ratio]
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
            tmpG = self.rfft_object_coarse(G.values) / self.coarse_prob.init
            tmpF = np.zeros(self.fine_prob.init // 2 + 1, dtype=np.complex128)
            halfG = int(self.coarse_prob.init / 2)
            tmpF[0: halfG] = tmpG[0: halfG]
            tmpF[-1] = tmpG[-1]
            F.values[:] = self.irfft_object_fine(tmpF, normalise_idft=False)
        elif isinstance(G, rhs_imex_mesh):
            F = rhs_imex_mesh(G)
            tmpG_impl = self.rfft_object_coarse(G.impl.values) / self.coarse_prob.init
            tmpF_impl = np.zeros(self.fine_prob.init // 2 + 1, dtype=np.complex128)
            halfG = int(self.coarse_prob.init / 2)
            tmpF_impl[0: halfG] = tmpG_impl[0: halfG]
            tmpF_impl[-1] = tmpG_impl[-1]
            F.impl.values[:] = self.irfft_object_fine(tmpF_impl, normalise_idft=False)
            tmpG_expl = self.rfft_object_coarse(G.expl.values) / self.coarse_prob.init
            tmpF_expl = np.zeros(self.fine_prob.init // 2 + 1, dtype=np.complex128)
            halfG = int(self.coarse_prob.init / 2)
            tmpF_expl[0: halfG] = tmpG_expl[0: halfG]
            tmpF_expl[-1] = tmpG_expl[-1]
            F.expl.values[:] = self.irfft_object_fine(tmpF_expl, normalise_idft=False)
        else:
            raise TransferError('Unknown data type, got %s' % type(G))
        return F
