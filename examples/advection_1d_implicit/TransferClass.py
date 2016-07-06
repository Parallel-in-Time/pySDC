from __future__ import division
import numpy as np

from pySDC.Transfer import transfer
from pySDC.datatype_classes.mesh import mesh, rhs_imex_mesh
import pySDC.Plugins.transfer_helper as th

# FIXME: extend this to ndarrays
class mesh_to_mesh_1d_periodic(transfer):
    """
    Custon transfer class, implements Transfer.py

    This implementation can restrict and prolong between 1d meshes via matrix-vector products

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

        fine_grid = np.array([i * fine_level.prob.dx for i in range(fine_level.prob.nvars)])
        coarse_grid = np.array([i * coarse_level.prob.dx for i in range(coarse_level.prob.nvars)])

        # if number of variables is the same on both levels, Rspace and Pspace are identity
        if self.init_c == self.init_f:
            self.Rspace = np.eye(self.init_c)
        # assemble restriction as transpose of interpolation
        else:

            if params['rorder'] == 1:

                self.Rspace = th.restriction_matrix_1d(fine_grid, coarse_grid, k=1, periodic=True)

            else:

                self.Rspace = 0.5 * th.interpolation_matrix_1d(fine_grid, coarse_grid, k=params['rorder'], periodic=True).T

        # if number of variables is the same on both levels, Rspace and Pspace are identity
        if self.init_f == self.init_c:
            self.Pspace = np.eye(self.init_f)
        else:
            self.Pspace = th.interpolation_matrix_1d(fine_grid, coarse_grid, k=params['iorder'], periodic=True)
        # print(self.Rspace.todense())
        # print(self.Pspace.todense())
        # exit()
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