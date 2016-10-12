from __future__ import division

import numpy as np
import scipy.sparse as sp

import pySDC.plugins.transfer_helper as th
from pySDC.SpaceTransfer import space_transfer


# FIXME: extend this to ndarrays
class mesh_to_mesh_1d_dirichlet(space_transfer):
    """
    Custon base_transfer class, implements Transfer.py

    This implementation can restrict and prolong between 1d meshes with dirichlet-0 boundaries via matrix-vector products

    Attributes:
        fine: reference to the fine level
        coarse: reference to the coarse level
        init_f: number of variables on the fine level (whatever init represents there)
        init_c: number of variables on the coarse level (whatever init represents there)
        Rspace: spatial restriction matrix, dim. Nf x Nc
        Pspace: spatial prolongation matrix, dim. Nc x Nf
    """

    def __init__(self,fine_prob,coarse_prob,params):
        """
        Initialization routine
        Args:
            fine_level: fine level connected with the base_transfer operations (passed to parent)
            coarse_level: coarse level connected with the base_transfer operations (passed to parent)
            params: parameters for the base_transfer operators
        """

        assert 'rorder' in params
        assert 'iorder' in params

        # invoke super initialization
        super(mesh_to_mesh_1d_dirichlet, self).__init__(fine_prob, coarse_prob, params)

        fine_grid = np.array([(i + 1) * self.fine_prob.dx for i in range(self.fine_prob.params.nvars)])
        coarse_grid = np.array([(i + 1) * self.coarse_prob.dx for i in range(self.coarse_prob.params.nvars)])

        # if number of variables is the same on both levels, Rspace and Pspace are identity
        if self.coarse_prob.params.nvars == self.fine_prob.params.nvars:
            self.Rspace = sp.eye(self.coarse_prob.params.nvars)
        # assemble restriction as transpose of interpolation
        else:
            self.Rspace = 0.5 * th.interpolation_matrix_1d_dirichlet_null(fine_grid, coarse_grid, k=self.params.rorder).T

        # if number of variables is the same on both levels, Rspace and Pspace are identity
        if self.coarse_prob.params.nvars == self.fine_prob.params.nvars:
            self.Pspace = sp.eye(self.fine_prob.params.nvars)
        else:
            self.Pspace = th.interpolation_matrix_1d_dirichlet_null(fine_grid, coarse_grid, k=self.params.iorder)

        pass

    def restrict(self,F):
        """
        Restriction implementation

        Args:
            F: the fine level data (easier to access than via the fine attribute)
        """

        u_coarse = None
        if isinstance(F,self.fine_prob.dtype_u):
            u_coarse = self.coarse_prob.dtype_u(self.coarse_prob.init,val=0)
            u_coarse.values = self.Rspace.dot(F.values)
        elif isinstance(F,self.fine_prob.dtype_f):
            u_coarse = self.coarse_prob.dtype_f(self.coarse_prob.init)
            u_coarse.impl.values = self.Rspace.dot(F.impl.values)
            u_coarse.expl.values = self.Rspace.dot(F.expl.values)

        return u_coarse

    def prolong(self,G):
        """
        Prolongation implementation

        Args:
            G: the coarse level data (easier to access than via the coarse attribute)
        """
        u_fine = None
        if isinstance(G,self.coarse_prob.dtype_u):
            u_fine = self.fine_prob.dtype_u(self.fine_prob.init,val=0)
            u_fine.values = self.Pspace.dot(G.values)
        elif isinstance(G,self.coarse_prob.dtype_f):
            u_fine = self.fine_prob.dtype_f(self.fine_prob.init)
            u_fine.impl.values = self.Pspace.dot(G.impl.values)
            u_fine.expl.values = self.Pspace.dot(G.expl.values)

        return u_fine