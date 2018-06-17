from __future__ import division

import numpy as np
import scipy.sparse as sp

import pySDC.helpers.transfer_helper as th
from pySDC.core.SpaceTransfer import space_transfer
from pySDC.core.Errors import TransferError

from pySDC.implementations.datatype_classes.mesh import mesh, rhs_imex_mesh, rhs_comp2_mesh


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

        if self.params.rorder % 2 != 0:
            raise TransferError('Need even order for restriction')

        if self.params.iorder % 2 != 0:
            raise TransferError('Need even order for interpolation')

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

            # if number of variables is the same on both levels, Rspace and Pspace are identity
            if self.coarse_prob.params.nvars == self.fine_prob.params.nvars:
                self.Rspace = sp.eye(self.coarse_prob.params.nvars)
                self.Pspace = sp.eye(self.fine_prob.params.nvars)
            # assemble restriction as transpose of interpolation
            else:

                if not self.params.periodic:
                    fine_grid = np.array([(i + 1) * self.fine_prob.dx for i in range(self.fine_prob.params.nvars)])
                    coarse_grid = np.array(
                        [(i + 1) * self.coarse_prob.dx for i in range(self.coarse_prob.params.nvars)])
                else:
                    fine_grid = np.array([i * self.fine_prob.dx for i in range(self.fine_prob.params.nvars)])
                    coarse_grid = np.array([i * self.coarse_prob.dx for i in range(self.coarse_prob.params.nvars)])

                self.Pspace = th.interpolation_matrix_1d(fine_grid, coarse_grid, k=self.params.iorder,
                                                         periodic=self.params.periodic,
                                                         equidist_nested=self.params.equidist_nested)
                if self.params.rorder > 0:
                    restr_factor = 0.5
                else:
                    restr_factor = 1.0

                if self.params.iorder == self.params.rorder:

                    self.Rspace = restr_factor * self.Pspace.T

                else:

                    self.Rspace = restr_factor * \
                        th.interpolation_matrix_1d(fine_grid, coarse_grid, k=self.params.rorder,
                                                   periodic=self.params.periodic,
                                                   equidist_nested=self.params.equidist_nested).T

        # we have an n-d problem
        else:

            Rspace = []
            Pspace = []
            for i in range(len(self.fine_prob.params.nvars)):

                # if number of variables is the same on both levels, Rspace and Pspace are identity
                if self.coarse_prob.params.nvars == self.fine_prob.params.nvars:
                    Rspace.append(sp.eye(self.coarse_prob.params.nvars[i]))
                    Pspace.append(sp.eye(self.fine_prob.params.nvars[i]))
                # assemble restriction as transpose of interpolation
                else:

                    if not self.params.periodic:
                        fine_grid = np.array(
                            [(j + 1) * self.fine_prob.dx for j in range(self.fine_prob.params.nvars[i])])
                        coarse_grid = np.array(
                            [(j + 1) * self.coarse_prob.dx for j in range(self.coarse_prob.params.nvars[i])])
                    else:
                        fine_grid = np.array([j * self.fine_prob.dx for j in range(self.fine_prob.params.nvars[i])])
                        coarse_grid = np.array(
                            [j * self.coarse_prob.dx for j in range(self.coarse_prob.params.nvars[i])])

                    Pspace.append(th.interpolation_matrix_1d(fine_grid, coarse_grid, k=self.params.iorder,
                                                             periodic=self.params.periodic,
                                                             equidist_nested=self.params.equidist_nested))
                    if self.params.rorder > 0:
                        restr_factor = 0.5
                    else:
                        restr_factor = 1.0

                    if self.params.iorder == self.params.rorder:

                        Rspace.append(restr_factor * Pspace[-1].T)

                    else:

                        Rspace.append(restr_factor *
                                      th.interpolation_matrix_1d(fine_grid, coarse_grid, k=self.params.rorder,
                                                                 periodic=self.params.periodic,
                                                                 equidist_nested=self.params.equidist_nested).T)

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
        if isinstance(F, mesh):
            F.values = F.values.flatten()
            G = F.apply_mat(self.Rspace)
            G.values = G.values.reshape(self.coarse_prob.params.nvars)
            F.values = F.values.reshape(self.fine_prob.params.nvars)
        elif isinstance(F, rhs_imex_mesh):
            F.impl.values = F.impl.values.flatten()
            F.expl.values = F.expl.values.flatten()
            G = F.apply_mat(self.Rspace)
            G.impl.values = G.impl.values.reshape(self.coarse_prob.params.nvars)
            G.expl.values = G.expl.values.reshape(self.coarse_prob.params.nvars)
            F.impl.values = F.impl.values.reshape(self.fine_prob.params.nvars)
            F.expl.values = F.expl.values.reshape(self.fine_prob.params.nvars)
        return G

    def prolong(self, G):
        """
        Prolongation implementation
        Args:
            G: the coarse level data (easier to access than via the coarse attribute)
        """
        if isinstance(G, mesh):
            G.values = G.values.flatten()
            F = G.apply_mat(self.Pspace)
            F.values = F.values.reshape(self.fine_prob.params.nvars)
            G.values = G.values.reshape(self.coarse_prob.params.nvars)
        elif isinstance(G, rhs_imex_mesh):
            G.impl.values = G.impl.values.flatten()
            G.expl.values = G.expl.values.flatten()
            F = G.apply_mat(self.Pspace)
            F.impl.values = F.impl.values.reshape(self.fine_prob.params.nvars)
            F.expl.values = F.expl.values.reshape(self.fine_prob.params.nvars)
            G.impl.values = G.impl.values.reshape(self.coarse_prob.params.nvars)
            G.expl.values = G.expl.values.reshape(self.coarse_prob.params.nvars)
        return F
