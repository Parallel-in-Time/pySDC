from __future__ import division
import dolfin as df

from pySDC.implementations.datatype_classes.fenics_mesh import fenics_mesh, rhs_fenics_mesh
from pySDC.core.SpaceTransfer import space_transfer
from pySDC.core.Errors import TransferError


class mesh_to_mesh_fenics(space_transfer):
    """
    This implementation can restrict and prolong between fenics meshes
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
        super(mesh_to_mesh_fenics, self).__init__(fine_prob, coarse_prob, params)

        pass

    def restrict(self, F):
        """
        Restriction implementation

        Args:
            F: the fine level data
        """
        if isinstance(F, fenics_mesh):
            u_coarse = fenics_mesh(df.interpolate(F.values, self.coarse_prob.init))
        elif isinstance(F, rhs_fenics_mesh):
            u_coarse = rhs_fenics_mesh(self.coarse_prob.init)
            u_coarse.impl.values = df.interpolate(F.impl.values, self.coarse_prob.init)
            u_coarse.expl.values = df.interpolate(F.expl.values, self.coarse_prob.init)
        else:
            raise TransferError('Unknown type of fine data, got %s' % type(F))

        return u_coarse

    def prolong(self, G):
        """
        Prolongation implementation

        Args:
            G: the coarse level data
        """
        if isinstance(G, fenics_mesh):
            u_fine = fenics_mesh(df.interpolate(G.values, self.fine_prob.init))
        elif isinstance(G, rhs_fenics_mesh):
            u_fine = rhs_fenics_mesh(self.fine_prob.init)
            u_fine.impl.values = df.interpolate(G.impl.values, self.fine_prob.init)
            u_fine.expl.values = df.interpolate(G.expl.values, self.fine_prob.init)
        else:
            raise TransferError('Unknown type of coarse data, got %s' % type(G))

        return u_fine
