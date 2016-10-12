from __future__ import division

import dolfin as df

from implementations.datatype_classes import rhs_fenics_mesh
from pySDC.implementations.datatype_classes import fenics_mesh
from pySDC.Transfer import transfer

class mesh_to_mesh_fenics(transfer):
    """
    Custon transfer class, implements Transfer.py

    This implementation can restrict and prolong between fenics meshes

    Attributes:
        fine: reference to the fine level
        coarse: reference to the coarse level
        init_f: number of variables on the fine level (whatever init represents there)
        init_c: number of variables on the coarse level (whatever init represents there)
    """

    def __init__(self,fine_level,coarse_level,params):
        """
        Initialization routine

        Args:
            fine_level: fine level connected with the transfer operations (passed to parent)
            coarse_level: coarse level connected with the transfer operations (passed to parent)
        """

        # invoke super initialization
        super(mesh_to_mesh_fenics,self).__init__(fine_level,coarse_level,params)

        pass

    def restrict_space(self,F):
        """
        Restriction implementation

        Args:
            F: the fine level data (easier to access than via the fine attribute)
        """

        if isinstance(F, fenics_mesh):
            u_coarse = fenics_mesh(self.init_c)
            u_coarse.values = df.interpolate(F.values,u_coarse.V)
        elif isinstance(F,rhs_fenics_mesh):
            u_coarse = rhs_fenics_mesh(self.init_c)
            u_coarse.impl.values = df.interpolate(F.impl.values,u_coarse.impl.V)
            u_coarse.expl.values = df.interpolate(F.expl.values,u_coarse.expl.V)

        return u_coarse

    def prolong_space(self,G):
        """
        Prolongation implementation

        Args:
            G: the coarse level data (easier to access than via the coarse attribute)
        """

        if isinstance(G, fenics_mesh):
            u_fine = fenics_mesh(self.init_f)
            u_fine.values = df.interpolate(G.values,u_fine.V)
        elif isinstance(G,rhs_fenics_mesh):
            u_fine = rhs_fenics_mesh(self.init_f)
            u_fine.impl.values = df.interpolate(G.impl.values,u_fine.impl.V)
            u_fine.expl.values = df.interpolate(G.expl.values,u_fine.expl.V)

        return u_fine