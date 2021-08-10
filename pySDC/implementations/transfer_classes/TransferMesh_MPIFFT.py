from pySDC.core.Errors import TransferError
from pySDC.core.SpaceTransfer import space_transfer
from pySDC.implementations.datatype_classes.mesh import mesh, imex_mesh
from mpi4py_fft import PFFT, newDistArray


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

        self.fft_pad = PFFT(self.coarse_prob.params.comm, Nc, padding=self.ratio, axes=axes,
                            dtype=self.coarse_prob.fft.dtype(False),
                            slab=True)

    def restrict(self, F):
        """
        Restriction implementation

        Args:
            F: the fine level data (easier to access than via the fine attribute)
        """
        if isinstance(F, mesh):
            if self.spectral:
                G = self.coarse_prob.dtype_u(self.coarse_prob.init)
                if hasattr(self.fine_prob, 'ncomp'):
                    for i in range(self.fine_prob.ncomp):
                        tmpF = newDistArray(self.fine_prob.fft, False)
                        tmpF = self.fine_prob.fft.backward(F[..., i], tmpF)
                        tmpG = tmpF[::int(self.ratio[0]), ::int(self.ratio[1])]
                        G[..., i] = self.coarse_prob.fft.forward(tmpG, G[..., i])
                else:
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
        if isinstance(G, mesh):
            if self.spectral:
                F = self.fine_prob.dtype_u(self.fine_prob.init)
                if hasattr(self.fine_prob, 'ncomp'):
                    for i in range(self.fine_prob.ncomp):
                        tmpF = self.fft_pad.backward(G[..., i])
                        F[..., i] = self.fine_prob.fft.forward(tmpF, F[..., i])
                else:
                    tmpF = self.fft_pad.backward(G)
                    F[:] = self.fine_prob.fft.forward(tmpF, F)
            else:
                F = self.fine_prob.dtype_u(self.fine_prob.init)
                if hasattr(self.fine_prob, 'ncomp'):
                    for i in range(self.fine_prob.ncomp):
                        G_hat = self.coarse_prob.fft.forward(G[..., i])
                        F[..., i] = self.fft_pad.backward(G_hat, F[..., i])
                else:
                    G_hat = self.coarse_prob.fft.forward(G)
                    F[:] = self.fft_pad.backward(G_hat, F)
        elif isinstance(G, imex_mesh):
            if self.spectral:
                F = self.fine_prob.dtype_f(self.fine_prob.init)
                if hasattr(self.fine_prob, 'ncomp'):
                    for i in range(self.fine_prob.ncomp):
                        tmpF = self.fft_pad.backward(G.impl[..., i])
                        F.impl[..., i] = self.fine_prob.fft.forward(tmpF, F.impl[..., i])
                        tmpF = self.fft_pad.backward(G.expl[..., i])
                        F.expl[..., i] = self.fine_prob.fft.forward(tmpF, F.expl[..., i])
                else:
                    tmpF = self.fft_pad.backward(G.impl)
                    F.impl[:] = self.fine_prob.fft.forward(tmpF, F.impl)
                    tmpF = self.fft_pad.backward(G.expl)
                    F.expl[:] = self.fine_prob.fft.forward(tmpF, F.expl)
            else:
                F = self.fine_prob.dtype_f(self.fine_prob.init)
                if hasattr(self.fine_prob, 'ncomp'):
                    for i in range(self.fine_prob.ncomp):
                        G_hat = self.coarse_prob.fft.forward(G.impl[..., i])
                        F.impl[..., i] = self.fft_pad.backward(G_hat, F.impl[..., i])
                        G_hat = self.coarse_prob.fft.forward(G.expl[..., i])
                        F.expl[..., i] = self.fft_pad.backward(G_hat, F.expl[..., i])
                else:
                    G_hat = self.coarse_prob.fft.forward(G.impl)
                    F.impl[:] = self.fft_pad.backward(G_hat, F.impl)
                    G_hat = self.coarse_prob.fft.forward(G.expl)
                    F.expl[:] = self.fft_pad.backward(G_hat, F.expl)
        else:
            raise TransferError('Unknown data type, got %s' % type(G))

        return F
