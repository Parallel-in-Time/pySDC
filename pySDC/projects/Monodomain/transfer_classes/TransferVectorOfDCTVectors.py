from pySDC.core.SpaceTransfer import space_transfer
from pySDC.projects.Monodomain.transfer_classes.Transfer_DCT_Vector import DCT_to_DCT
from pySDC.projects.Monodomain.datatype_classes.VectorOfVectors import VectorOfVectors, IMEXEXP_VectorOfVectors


class TransferVectorOfDCTVectors(space_transfer):
    """
    This implementation can restrict and prolong between super vectors
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
        super(TransferVectorOfDCTVectors, self).__init__(fine_prob, coarse_prob, params)

        self.DCT_to_DCT = DCT_to_DCT(fine_prob, coarse_prob, params)

    def restrict(self, F):
        """
        Restriction implementation

        Args:
            F: the fine level data
        """
        u_coarse = VectorOfVectors(
            init=self.coarse_prob.init,
            val=0.0,
            type_sub_vector=self.coarse_prob.vector_type,
            size=self.coarse_prob.size,
        )

        for i in range(u_coarse.size):
            u_coarse.val_list[i].values[:] = self.DCT_to_DCT.restrict(F[i]).values

        return u_coarse

    def prolong(self, G):
        """
        Prolongation implementation

        Args:
            G: the coarse level data
        """
        if isinstance(G, VectorOfVectors):
            u_fine = VectorOfVectors(
                init=self.fine_prob.init, val=0.0, type_sub_vector=self.fine_prob.vector_type, size=self.fine_prob.size
            )
            for i in range(u_fine.size):
                u_fine.val_list[i].values[:] = self.DCT_to_DCT.prolong(G[i]).values
        elif isinstance(G, IMEXEXP_VectorOfVectors):
            u_fine = IMEXEXP_VectorOfVectors(
                init=self.fine_prob.init, val=0.0, type_sub_vector=self.fine_prob.vector_type, size=self.fine_prob.size
            )
            for i in range(u_fine.size):
                u_fine.impl.val_list[i].values[:] = self.DCT_to_DCT.prolong(G.impl[i]).values
                u_fine.expl.val_list[i].values[:] = self.DCT_to_DCT.prolong(G.expl[i]).values
                u_fine.exp.val_list[i].values[:] = self.DCT_to_DCT.prolong(G.exp[i]).values

        return u_fine
