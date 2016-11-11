from __future__ import division

import numpy as np

import pySDC_core.plugins.transfer_helper as th
from pySDC_implementations.datatype_classes import mesh, rhs_imex_mesh
from pySDC_core.Transfer import transfer


# FIXME: extend this to ndarrays
# FIXME: extend this to ndarrays
class mesh_to_mesh_1d_periodic(transfer):
    """
    Custon transfer class, implements Transfer.py

    This implementation can restrict and prolong between 1d meshes, using weigthed restriction and 7th-order prologation
    via matrix-vector multiplication.

    Attributes:
        fine: reference to the fine level
        coarse: reference to the coarse level
        init_f: number of variables on the fine level (whatever init represents there)
        init_c: number of variables on the coarse level (whatever init represents there)
        Rspace: spatial restriction matrix, dim. Nf x Nc
        Pspace: spatial prolongation matrix, dim. Nc x Nf
    """

    def __init__(self,fine_level,coarse_level,params):
        """
        Initialization routine

        Args:
            fine_level: fine level connected with the transfer operations (passed to parent)
            coarse_level: coarse level connected with the transfer operations (passed to parent)
            params: parameters for the transfer operators
        """

        # invoke super initialization
        super(mesh_to_mesh_1d_periodic,self).__init__(fine_level,coarse_level,params)

        xvals_fine = np.array([i*self.fine.prob.dx for i in range(self.init_f)])
        xvals_coarse = np.array([i*self.coarse.prob.dx for i in range(self.init_c)])
        right_end_point = 1.0

        # if number of variables is the same on both levels, Rspace and Pspace are identity
        if self.init_c == self.init_f:
            self.Rspace = np.eye(self.init_c)
            self.Pspace = np.eye(self.init_f)
        else:

            self.Pspace = th.interpolation_matrix_1d(xvals_fine,xvals_coarse,self.params.iorder,periodic=True,T=right_end_point)
            self.Rspace = th.restriction_matrix_1d(xvals_fine,xvals_coarse,self.params.rorder,periodic=True,T=right_end_point)
        pass

    def restrict_space(self,F):
        """
        Restriction implementation

        Args:
            F: the fine level data (easier to access than via the fine attribute)
        """

        if isinstance(F,mesh):
            u_coarse = mesh(self.init_c,val=0)
            u_coarse.values = self.Rspace.dot(F.values)
        elif isinstance(F,rhs_imex_mesh):
            u_coarse = rhs_imex_mesh(self.init_c)
            u_coarse.impl.values = self.Rspace.dot(F.impl.values)
            u_coarse.expl.values = self.Rspace.dot(F.expl.values)

        return u_coarse

    def prolong_space(self,G):
        """
        Prolongation implementation

        Args:
            G: the coarse level data (easier to access than via the coarse attribute)
        """

        if isinstance(G,mesh):
            u_fine = mesh(self.init_c,val=0)
            u_fine.values = self.Pspace.dot(G.values)
        elif isinstance(G,rhs_imex_mesh):
            u_fine = rhs_imex_mesh(self.init_c)
            u_fine.impl.values = self.Pspace.dot(G.impl.values)
            u_fine.expl.values = self.Pspace.dot(G.expl.values)

        return u_fine