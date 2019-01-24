
import scipy.sparse as sp

from pySDC.core.Errors import TransferError
from pySDC.core.SpaceTransfer import space_transfer
from pySDC.implementations.datatype_classes.complex_mesh import mesh as cmesh
from pySDC.implementations.datatype_classes.mesh import mesh, rhs_imex_mesh


class mesh_to_mesh(space_transfer):
    """
    Custon base_transfer class, implements Transfer.py

    This implementation can restrict and prolong between nd meshes with dirichlet-0 or periodic boundaries
    via matrix-vector products

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
        super(mesh_to_mesh, self).__init__(fine_prob, coarse_prob, params)

        self.Rspace = sp.eye(self.coarse_prob.params.nvars)
        self.Pspace = sp.eye(self.fine_prob.params.nvars)

    def restrict(self, F):
        """
        Restriction implementation

        Args:
            F: the fine level data (easier to access than via the fine attribute)
        """
        if isinstance(F, mesh):
            G = mesh(F)
        elif isinstance(F, cmesh):
            G = cmesh(F)
        elif isinstance(F, rhs_imex_mesh):
            G = rhs_imex_mesh(F)
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
            F = mesh(G)
        elif isinstance(G, cmesh):
            F = cmesh(G)
        elif isinstance(G, rhs_imex_mesh):
            F = rhs_imex_mesh(G)
        else:
            raise TransferError('Unknown data type, got %s' % type(G))
        return F
