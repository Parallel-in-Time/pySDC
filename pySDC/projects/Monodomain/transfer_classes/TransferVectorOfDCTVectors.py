from pySDC.core.space_transfer import SpaceTransfer
from pySDC.projects.Monodomain.transfer_classes.Transfer_DCT_Vector import DCT_to_DCT
from pySDC.implementations.datatype_classes.mesh import mesh
from pySDC.projects.Monodomain.datatype_classes.my_mesh import imexexp_mesh


class TransferVectorOfDCTVectors(SpaceTransfer):
    """
    This implementation can restrict and prolong VectorOfVectors
    """

    def __init__(self, fine_prob, coarse_prob, params):
        # invoke super initialization
        super(TransferVectorOfDCTVectors, self).__init__(fine_prob, coarse_prob, params)

        self.DCT_to_DCT = DCT_to_DCT(fine_prob, coarse_prob, params)

    def restrict(self, F):

        u_coarse = mesh(self.coarse_prob.init)

        for i in range(self.coarse_prob.size):
            u_coarse[i][:] = self.DCT_to_DCT.restrict(F[i])

        return u_coarse

    def prolong(self, G):

        if isinstance(G, imexexp_mesh):
            u_fine = imexexp_mesh(self.fine_prob.init)
            for i in range(self.fine_prob.size):
                u_fine.impl[i][:] = self.DCT_to_DCT.prolong(G.impl[i])
                u_fine.expl[i][:] = self.DCT_to_DCT.prolong(G.expl[i])
                u_fine.exp[i][:] = self.DCT_to_DCT.prolong(G.exp[i])
        elif isinstance(G, mesh):
            u_fine = mesh(self.fine_prob.init)
            for i in range(self.fine_prob.size):
                u_fine[i][:] = self.DCT_to_DCT.prolong(G[i])

        return u_fine
