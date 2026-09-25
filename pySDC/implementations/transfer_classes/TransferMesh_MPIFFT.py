from pySDC.core.errors import TransferError
from pySDC.core.space_transfer import SpaceTransfer
from pySDC.helpers.fft import newDistArray

from pySDC.helpers.fft_helper import PFFT


class fft_to_fft(SpaceTransfer):
    """
    Custom base_transfer class, implements Transfer.py

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
        super().__init__(fine_prob, coarse_prob, params)

        assert self.fine_prob.spectral == self.coarse_prob.spectral

        self.spectral = self.fine_prob.spectral

        Nf = list(self.fine_prob.fft.global_shape())
        Nc = list(self.coarse_prob.fft.global_shape())
        self.ratio = [int(nf / nc) for nf, nc in zip(Nf, Nc, strict=True)]
        # slice for injection, one entry per dimension rather than hardcoded for 2d
        self.injection = tuple(slice(None, None, r) for r in self.ratio)
        axes = tuple(range(len(Nf)))

        fft_args = {}
        useGPU = 'cupy' in self.fine_prob.dtype_u.__name__.lower()
        if useGPU:
            fft_args['backend'] = 'cupy'
            fft_args['comm_backend'] = 'NCCL'

        self.fft_pad = PFFT(
            self.coarse_prob.comm,
            Nc,
            padding=self.ratio,
            axes=axes,
            dtype=self.coarse_prob.fft.dtype(False),
            slab=True,
            **fft_args,
        )

    def restrict(self, F):
        """
        Restriction implementation

        Args:
            F: the fine level data (easier to access than via the fine attribute)
        """
        G = type(F)(self.coarse_prob.init)

        def _restrict(fine, coarse):
            if self.spectral:
                if hasattr(self.fine_prob, 'ncomp'):
                    for i in range(self.fine_prob.ncomp):
                        if fine.shape[-1] == self.fine_prob.ncomp:
                            tmpF = newDistArray(self.fine_prob.fft, False)
                            tmpF = self.fine_prob.fft.backward(fine[..., i], tmpF)
                            tmpG = tmpF[self.injection]
                            coarse[..., i] = self.coarse_prob.fft.forward(tmpG, coarse[..., i])
                        elif fine.shape[0] == self.fine_prob.ncomp:
                            tmpF = newDistArray(self.fine_prob.fft, False)
                            tmpF = self.fine_prob.fft.backward(fine[i, ...], tmpF)
                            tmpG = tmpF[self.injection]
                            coarse[i, ...] = self.coarse_prob.fft.forward(tmpG, coarse[i, ...])
                        else:
                            raise TransferError('Don\'t know how to restrict for this problem with multiple components')
                else:
                    tmpF = self.fine_prob.fft.backward(fine)
                    tmpG = tmpF[self.injection]
                    coarse[:] = self.coarse_prob.fft.forward(tmpG, coarse)
            elif hasattr(self.fine_prob, 'ncomp') and fine.shape[0] == self.fine_prob.ncomp:
                # the component axis comes first here, so inject on the spatial axes behind it. A
                # trailing component axis needs no special case: the slice simply does not reach it.
                coarse[:] = fine[(slice(None),) + self.injection]
            else:
                coarse[:] = fine[self.injection]

        if hasattr(type(F), 'components'):
            for comp in F.components:
                _restrict(getattr(F, comp), getattr(G, comp))
        elif type(F).__name__ in ['mesh', 'cupy_mesh']:
            _restrict(F, G)
        else:
            raise TransferError('Wrong data type for restriction, got %s' % type(F))

        return G

    def prolong(self, G):
        """
        Prolongation implementation

        Args:
            G: the coarse level data (easier to access than via the coarse attribute)
        """
        F = type(G)(self.fine_prob.init)

        def _prolong(coarse, fine):
            if self.spectral:
                if hasattr(self.fine_prob, 'ncomp'):
                    for i in range(self.fine_prob.ncomp):
                        if coarse.shape[-1] == self.fine_prob.ncomp:
                            tmpF = self.fft_pad.backward(coarse[..., i])
                            fine[..., i] = self.fine_prob.fft.forward(tmpF, fine[..., i])
                        elif coarse.shape[0] == self.fine_prob.ncomp:
                            tmpF = self.fft_pad.backward(coarse[i, ...])
                            fine[i, ...] = self.fine_prob.fft.forward(tmpF, fine[i, ...])
                        else:
                            raise TransferError('Don\'t know how to prolong for this problem with multiple components')

                else:
                    tmpF = self.fft_pad.backward(coarse)
                    fine[:] = self.fine_prob.fft.forward(tmpF, fine)
            else:
                if hasattr(self.fine_prob, 'ncomp'):
                    for i in range(self.fine_prob.ncomp):
                        if coarse.shape[-1] == self.fine_prob.ncomp:
                            G_hat = self.coarse_prob.fft.forward(coarse[..., i])
                            fine[..., i] = self.fft_pad.backward(G_hat, fine[..., i])
                        elif coarse.shape[0] == self.fine_prob.ncomp:
                            G_hat = self.coarse_prob.fft.forward(coarse[i, ...])
                            fine[i, ...] = self.fft_pad.backward(G_hat, fine[i, ...])
                        else:
                            raise TransferError('Don\'t know how to prolong for this problem with multiple components')
                else:
                    G_hat = self.coarse_prob.fft.forward(coarse)
                    fine[:] = self.fft_pad.backward(G_hat, fine)

        if hasattr(type(F), 'components'):
            for comp in F.components:
                _prolong(getattr(G, comp), getattr(F, comp))
        elif type(G).__name__ in ['mesh', 'cupy_mesh']:
            _prolong(G, F)

        else:
            raise TransferError('Unknown data type, got %s' % type(G))

        return F
