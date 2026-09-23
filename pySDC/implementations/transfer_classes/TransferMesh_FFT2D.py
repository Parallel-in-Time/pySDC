from scipy.signal import resample

from pySDC.core.errors import TransferError
from pySDC.core.space_transfer import SpaceTransfer


class mesh_to_mesh_fft2d(SpaceTransfer):
    """
    Custon base_transfer class, implements Transfer.py

    This implementation can restrict and prolong between 2d meshes with FFT for periodic boundaries

    Attributes:
        ratio: refinement factor between the two meshes
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
        G = type(F)(self.coarse_prob.init, val=0.0)

        def _restrict(fine, coarse):
            coarse[:] = fine[:: self.ratio, :: self.ratio]

        # note that a `MultiComponentMesh` is also an instance of `mesh`, so ask for the components
        # rather than for the type
        if hasattr(type(F), 'components'):
            for comp in F.components:
                _restrict(getattr(F, comp), getattr(G, comp))
        elif type(F).__name__ == 'mesh':
            _restrict(F, G)
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

        def _prolong(coarse, fine):
            # Fourier interpolation along both axes. `resample` also gets the normalisation and the
            # splitting of the Nyquist mode right, which hand-rolled zero padding of the spectrum
            # only did for a refinement factor of two.
            fine[:] = resample(resample(coarse, fine.shape[0], axis=0), fine.shape[1], axis=1)

        if hasattr(type(G), 'components'):
            for comp in G.components:
                _prolong(getattr(G, comp), getattr(F, comp))
        elif type(G).__name__ == 'mesh':
            _prolong(G, F)
        else:
            raise TransferError('Unknown data type, got %s' % type(G))
        return F
