import scipy.fft as fft

from pySDC.core.SpaceTransfer import space_transfer
from pySDC.projects.Monodomain.datatype_classes.DCT_Vector import DCT_Vector


class DCT_to_DCT(space_transfer):
    """
    Class to transfer data between two meshes using DCT.
    Restriction is performed by zeroing out high frequency modes, while prolongation is done by zero-padding.

    Arguments:
    ----------
            fine_prob: fine problem
            coarse_prob: coarse problem
            params: parameters for the transfer operators
    """

    def __init__(self, fine_prob, coarse_prob, params):

        # invoke super initialization
        super(DCT_to_DCT, self).__init__(fine_prob, coarse_prob, params)

        self.norm = "forward"

        self.fine_shape = self.fine_prob.parabolic.shape
        self.coarse_shape = self.coarse_prob.parabolic.shape

        if self.fine_shape == self.coarse_shape:
            self.same_grid = True
        else:
            self.same_grid = False

    def restrict(self, F):
        """
        Restriction opeartor
        Args:
            F: the fine level data (easier to access than via the fine attribute)
        """

        if self.same_grid:
            G = F.copy()
        else:
            G = DCT_Vector(self.coarse_prob.init)
            G.values[:] = fft.idctn(
                fft.dctn(F.values.reshape(self.fine_shape), norm=self.norm), s=self.coarse_shape, norm=self.norm
            ).ravel()

        return G

    def prolong(self, G):
        """
        Prolongation opeartor
        Args:
            G: the coarse level data (easier to access than via the coarse attribute)
        """

        if self.same_grid:
            F = G.copy()
        else:
            F = DCT_Vector(self.fine_prob.init)
            F.values[:] = fft.idctn(
                fft.dctn(G.values.reshape(self.coarse_shape), norm=self.norm), s=self.fine_shape, norm=self.norm
            ).ravel()

        return F
