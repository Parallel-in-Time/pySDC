import dolfinx as df
from petsc4py import PETSc
from pySDC.core.Errors import TransferError
from pySDC.core.SpaceTransfer import space_transfer
from pySDC.projects.Monodomain_NEW.datatype_classes.VectorOfVectors import VectorOfVectors, IMEXEXP_VectorOfVectors


class TransferVectorOfVectors(space_transfer):
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
        super(TransferVectorOfVectors, self).__init__(fine_prob, coarse_prob, params)

        pass

    def restrict(self, F):
        """
        Restriction implementation

        Args:
            F: the fine level data
        """
        u_coarse = type(F)(init=self.coarse_prob.init, val=0.0, type_sub_vector=self.coarse_prob.vector_type, size=self.coarse_prob.size)
        F.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        u_coarse.restrict(F)

        return u_coarse

    def prolong(self, G):
        """
        Prolongation implementation

        Args:
            G: the coarse level data
        """
        u_fine = type(G)(init=self.fine_prob.init, val=0.0, type_sub_vector=self.fine_prob.vector_type, size=self.fine_prob.size)
        G.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        u_fine.prolong(G)

        return u_fine
