from __future__ import division

import numpy as np
import scipy.sparse as sp

import pySDC.plugins.transfer_helper as th
from pySDC.SpaceTransfer import space_transfer


class mesh_to_mesh(space_transfer):
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
        super(mesh_to_mesh, self).__init__(fine_prob, coarse_prob, params)

        if type(self.fine_prob.params.nvars) is tuple:
            assert type(self.coarse_prob.params.nvars) is tuple
            assert len(self.fine_prob.params.nvars) == len(self.coarse_prob.params.nvars)
        elif type(self.fine_prob.params.nvars) is int:
            assert type(self.coarse_prob.params.nvars) is int
        else:
            print("ERROR: unknow type of nvars for transfer, got %s" %self.fine_prob.params.nvars)
            exit()

        if type(self.fine_prob.params.nvars) is int:

            if not self.params.periodic:
                fine_grid = np.array([(i+1) * self.fine_prob.dx for i in range(self.fine_prob.params.nvars)])
                coarse_grid = np.array([(i+1)  * self.coarse_prob.dx for i in range(self.coarse_prob.params.nvars)])
            else:
                fine_grid = np.array([i * self.fine_prob.dx for i in range(self.fine_prob.params.nvars)])
                coarse_grid = np.array([i * self.coarse_prob.dx for i in range(self.coarse_prob.params.nvars)])

            # if number of variables is the same on both levels, Rspace and Pspace are identity
            if self.coarse_prob.params.nvars == self.fine_prob.params.nvars:
                self.Rspace = sp.eye(self.coarse_prob.params.nvars)
            # assemble restriction as transpose of interpolation
            else:
                self.Rspace = 0.5 * th.interpolation_matrix_1d(fine_grid, coarse_grid, k=self.params.rorder, periodic=self.params.periodic).T

            # if number of variables is the same on both levels, Rspace and Pspace are identity
            if self.coarse_prob.params.nvars == self.fine_prob.params.nvars:
                self.Pspace = sp.eye(self.fine_prob.params.nvars)
            else:
                self.Pspace = th.interpolation_matrix_1d(fine_grid, coarse_grid, k=self.params.iorder, periodic=self.params.periodic)

        else:

            Rspace = []
            Pspace = []
            for i in range(len(self.fine_prob.params.nvars)):

                if not self.params.periodic:
                    fine_grid = np.array([(j+1) * self.fine_prob.dx for j in range(self.fine_prob.params.nvars[i])])
                    coarse_grid = np.array([(j+1) * self.coarse_prob.dx for j in range(self.coarse_prob.params.nvars[i])])
                else:
                    fine_grid = np.array([j * self.fine_prob.dx for j in range(self.fine_prob.params.nvars[i])])
                    coarse_grid = np.array([j * self.coarse_prob.dx for j in range(self.coarse_prob.params.nvars[i])])

                # if number of variables is the same on both levels, Rspace and Pspace are identity
                if self.coarse_prob.params.nvars == self.fine_prob.params.nvars:
                    Rspace.append(sp.eye(self.coarse_prob.params.nvars[i]))
                # assemble restriction as transpose of interpolation
                else:
                    Rspace.append(0.5 * th.interpolation_matrix_1d(fine_grid, coarse_grid, k=self.params.iorder, periodic=self.params.periodic).T)

                # if number of variables is the same on both levels, Rspace and Pspace are identity
                if self.coarse_prob.params.nvars == self.fine_prob.params.nvars:
                    Pspace.append(sp.eye(self.fine_prob.params.nvars[i]))
                else:
                    Pspace.append(th.interpolation_matrix_1d(fine_grid, coarse_grid, k=self.params.iorder, periodic=self.params.periodic))

            self.Pspace = Pspace[0]
            for i in range(1,len(Pspace)):
                self.Pspace = sp.kron(self.Pspace,Pspace[i],format='csc')

            self.Rspace = Rspace[0]
            for i in range(1, len(Rspace)):
                self.Rspace = sp.kron(self.Rspace, Rspace[i], format='csc')

    def restrict(self,F):
        """
        Restriction implementation

        Args:
            F: the fine level data (easier to access than via the fine attribute)
        """
        return F.apply_mat(self.Rspace)

    def prolong(self,G):
        """
        Prolongation implementation

        Args:
            G: the coarse level data (easier to access than via the coarse attribute)
        """
        return G.apply_mat(self.Pspace)


