from __future__ import division

import numpy as np
import scipy.sparse as sp

import pySDC.plugins.transfer_helper as th
from pySDC.SpaceTransfer import space_transfer
from pySDC.Errors import TransferError


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

        if 'iorder' not in params:
            raise TransferError('Need iorder parameter for spatial transfer')
        if 'rorder' not in params:
            raise TransferError('Need rorder parameter for spatial transfer')

        # invoke super initialization
        super(mesh_to_mesh, self).__init__(fine_prob, coarse_prob, params)

        if type(self.fine_prob.params.nvars) is tuple:
            if type(self.coarse_prob.params.nvars) is not tuple:
                raise TransferError('nvars parameter of coarse problem needs to be a tuple')
            if not len(self.fine_prob.params.nvars) == len(self.coarse_prob.params.nvars):
                raise TransferError('nvars parameter of fine and coarse level needs to have the same length')
        elif type(self.fine_prob.params.nvars) is int:
            if type(self.coarse_prob.params.nvars) is not int:
                raise TransferError('nvars parameter of coarse problem needs to be an int')
        else:
            raise TransferError("unknow type of nvars for transfer, got %s" % self.fine_prob.params.nvars)

        # we have a 1d problem
        if type(self.fine_prob.params.nvars) is int:

            if not self.params.periodic:
                fine_grid = np.array([(i + 1) * self.fine_prob.dx for i in range(self.fine_prob.params.nvars)])
                coarse_grid = np.array([(i + 1) * self.coarse_prob.dx for i in range(self.coarse_prob.params.nvars)])
            else:
                fine_grid = np.array([i * self.fine_prob.dx for i in range(self.fine_prob.params.nvars)])
                coarse_grid = np.array([i * self.coarse_prob.dx for i in range(self.coarse_prob.params.nvars)])

            # if number of variables is the same on both levels, Rspace and Pspace are identity
            if self.coarse_prob.params.nvars == self.fine_prob.params.nvars:
                self.Rspace = sp.eye(self.coarse_prob.params.nvars)
            # assemble restriction as transpose of interpolation
            else:
                self.Rspace = 0.5 * th.interpolation_matrix_1d(fine_grid, coarse_grid, k=self.params.rorder,
                                                               periodic=self.params.periodic).T

            # if number of variables is the same on both levels, Rspace and Pspace are identity
            if self.coarse_prob.params.nvars == self.fine_prob.params.nvars:
                self.Pspace = sp.eye(self.fine_prob.params.nvars)
            else:
                self.Pspace = th.interpolation_matrix_1d(fine_grid, coarse_grid, k=self.params.iorder,
                                                         periodic=self.params.periodic)
        # we have an n-d problem
        else:

            Rspace = []
            Pspace = []
            for i in range(len(self.fine_prob.params.nvars)):

                if not self.params.periodic:
                    fine_grid = np.array([(j + 1) * self.fine_prob.dx for j in range(self.fine_prob.params.nvars[i])])
                    coarse_grid = np.array(
                        [(j + 1) * self.coarse_prob.dx for j in range(self.coarse_prob.params.nvars[i])])
                else:
                    fine_grid = np.array([j * self.fine_prob.dx for j in range(self.fine_prob.params.nvars[i])])
                    coarse_grid = np.array([j * self.coarse_prob.dx for j in range(self.coarse_prob.params.nvars[i])])

                # if number of variables is the same on both levels, Rspace and Pspace are identity
                if self.coarse_prob.params.nvars == self.fine_prob.params.nvars:
                    Rspace.append(sp.eye(self.coarse_prob.params.nvars[i]))
                # assemble restriction as transpose of interpolation
                else:
                    Rspace.append(0.5 * th.interpolation_matrix_1d(fine_grid, coarse_grid, k=self.params.iorder,
                                                                   periodic=self.params.periodic).T)

                # if number of variables is the same on both levels, Rspace and Pspace are identity
                if self.coarse_prob.params.nvars == self.fine_prob.params.nvars:
                    Pspace.append(sp.eye(self.fine_prob.params.nvars[i]))
                else:
                    Pspace.append(th.interpolation_matrix_1d(fine_grid, coarse_grid, k=self.params.iorder,
                                                             periodic=self.params.periodic))
            # kronecker 1-d operators for n-d
            self.Pspace = Pspace[0]
            for i in range(1, len(Pspace)):
                self.Pspace = sp.kron(self.Pspace, Pspace[i], format='csc')

            self.Rspace = Rspace[0]
            for i in range(1, len(Rspace)):
                self.Rspace = sp.kron(self.Rspace, Rspace[i], format='csc')

    def restrict(self, F):
        """
        Restriction implementation

        Args:
            F: the fine level data (easier to access than via the fine attribute)
        """
        return F.apply_mat(self.Rspace)

    def prolong(self, G):
        """
        Prolongation implementation

        Args:
            G: the coarse level data (easier to access than via the coarse attribute)
        """
        return G.apply_mat(self.Pspace)
