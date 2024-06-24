import numpy as np
import scipy.sparse as sp

import pySDC.helpers.transfer_helper as th
from pySDC.core.errors import TransferError
from pySDC.core.space_transfer import SpaceTransfer


class mesh_to_mesh(SpaceTransfer):
    """
    Custom base_transfer class, implements Transfer.py

    This implementation can restrict and prolong between nd meshes with dirichlet-0 or periodic boundaries
    via matrix-vector products.

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
        super().__init__(fine_prob, coarse_prob, params)

        if self.params.rorder % 2 != 0:
            raise TransferError('Need even order for restriction')

        if self.params.iorder % 2 != 0:
            raise TransferError('Need even order for interpolation')

        if type(self.fine_prob.nvars) is tuple:
            if type(self.coarse_prob.nvars) is not tuple:
                raise TransferError('nvars parameter of coarse problem needs to be a tuple')
            if not len(self.fine_prob.nvars) == len(self.coarse_prob.nvars):
                raise TransferError('nvars parameter of fine and coarse level needs to have the same length')
        elif type(self.fine_prob.nvars) is int:
            if type(self.coarse_prob.nvars) is not int:
                raise TransferError('nvars parameter of coarse problem needs to be an int')
        else:
            raise TransferError("unknow type of nvars for transfer, got %s" % self.fine_prob.nvars)

        # we have a 1d problem
        if type(self.fine_prob.nvars) is int:
            # if number of variables is the same on both levels, Rspace and Pspace are identity
            if self.coarse_prob.nvars == self.fine_prob.nvars:
                self.Rspace = sp.eye(self.coarse_prob.nvars)
                self.Pspace = sp.eye(self.fine_prob.nvars)
            # assemble restriction as transpose of interpolation
            else:
                if not self.params.periodic:
                    fine_grid = np.array([(i + 1) * self.fine_prob.dx for i in range(self.fine_prob.nvars)])
                    coarse_grid = np.array([(i + 1) * self.coarse_prob.dx for i in range(self.coarse_prob.nvars)])
                else:
                    fine_grid = np.array([i * self.fine_prob.dx for i in range(self.fine_prob.nvars)])
                    coarse_grid = np.array([i * self.coarse_prob.dx for i in range(self.coarse_prob.nvars)])

                self.Pspace = th.interpolation_matrix_1d(
                    fine_grid,
                    coarse_grid,
                    k=self.params.iorder,
                    periodic=self.params.periodic,
                    equidist_nested=self.params.equidist_nested,
                )
                if self.params.rorder > 0:
                    restr_factor = 0.5
                else:
                    restr_factor = 1.0

                if self.params.iorder == self.params.rorder:
                    self.Rspace = restr_factor * self.Pspace.T

                else:
                    self.Rspace = (
                        restr_factor
                        * th.interpolation_matrix_1d(
                            fine_grid,
                            coarse_grid,
                            k=self.params.rorder,
                            periodic=self.params.periodic,
                            equidist_nested=self.params.equidist_nested,
                        ).T
                    )

        # we have an n-d problem
        else:
            Rspace = []
            Pspace = []
            for i in range(len(self.fine_prob.nvars)):
                # if number of variables is the same on both levels, Rspace and Pspace are identity
                if self.coarse_prob.nvars == self.fine_prob.nvars:
                    Rspace.append(sp.eye(self.coarse_prob.nvars[i]))
                    Pspace.append(sp.eye(self.fine_prob.nvars[i]))
                # assemble restriction as transpose of interpolation
                else:
                    if not self.params.periodic:
                        fine_grid = np.array([(j + 1) * self.fine_prob.dx for j in range(self.fine_prob.nvars[i])])
                        coarse_grid = np.array(
                            [(j + 1) * self.coarse_prob.dx for j in range(self.coarse_prob.nvars[i])]
                        )
                    else:
                        fine_grid = np.array([j * self.fine_prob.dx for j in range(self.fine_prob.nvars[i])])
                        coarse_grid = np.array([j * self.coarse_prob.dx for j in range(self.coarse_prob.nvars[i])])

                    Pspace.append(
                        th.interpolation_matrix_1d(
                            fine_grid,
                            coarse_grid,
                            k=self.params.iorder,
                            periodic=self.params.periodic,
                            equidist_nested=self.params.equidist_nested,
                        )
                    )
                    if self.params.rorder > 0:
                        restr_factor = 0.5
                    else:
                        restr_factor = 1.0

                    if self.params.iorder == self.params.rorder:
                        Rspace.append(restr_factor * Pspace[-1].T)

                    else:
                        mat = th.interpolation_matrix_1d(
                            fine_grid,
                            coarse_grid,
                            k=self.params.rorder,
                            periodic=self.params.periodic,
                            equidist_nested=self.params.equidist_nested,
                        ).T
                        Rspace.append(restr_factor * mat)

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
        G = type(F)(self.coarse_prob.init)

        def _restrict(fine, coarse):
            if hasattr(self.fine_prob, 'ncomp'):
                for i in range(self.fine_prob.ncomp):
                    if fine.shape[-1] == self.fine_prob.ncomp:
                        tmpF = fine[..., i].flatten()
                        tmpG = self.Rspace.dot(tmpF)
                        coarse[..., i] = tmpG.reshape(self.coarse_prob.nvars)
                    elif fine.shape[0] == self.fine_prob.ncomp:
                        tmpF = fine[i, ...].flatten()
                        tmpG = self.Rspace.dot(tmpF)
                        coarse[i, ...] = tmpG.reshape(self.coarse_prob.nvars)
                    else:
                        raise TransferError('Don\'t know how to restrict for this problem with multiple components')
            else:
                tmpF = fine.flatten()
                tmpG = self.Rspace.dot(tmpF)
                coarse[:] = tmpG.reshape(self.coarse_prob.nvars)

        if hasattr(type(F), 'components'):
            for comp in F.components:
                _restrict(F.__getattr__(comp), G.__getattr__(comp))
        elif type(F).__name__ == 'mesh':
            _restrict(F, G)
        else:
            raise TransferError('Wrong data type for restriction, got %s' % type(F))
        return G

    def prolong(self, G):
        """
        Prolongation implementation
        Args:
            G: the coarse level data (easier to access than via the coarse attribute)
        """
        F = type(G)(self.fine_prob.init)

        def _prolong(coarse, fine):
            if hasattr(self.fine_prob, 'ncomp'):
                for i in range(self.fine_prob.ncomp):
                    if coarse.shape[-1] == self.fine_prob.ncomp:
                        tmpG = coarse[..., i].flatten()
                        tmpF = self.Pspace.dot(tmpG)
                        fine[..., i] = tmpF.reshape(self.fine_prob.nvars)
                    elif coarse.shape[0] == self.fine_prob.ncomp:
                        tmpG = coarse[i, ...].flatten()
                        tmpF = self.Pspace.dot(tmpG)
                        fine[i, ...] = tmpF.reshape(self.fine_prob.nvars)
                    else:
                        raise TransferError('Don\'t know how to prolong for this problem with multiple components')
            else:
                tmpG = coarse.flatten()
                tmpF = self.Pspace.dot(tmpG)
                fine[:] = tmpF.reshape(self.fine_prob.nvars)
            return fine

        if hasattr(type(F), 'components'):
            for comp in G.components:
                _prolong(G.__getattr__(comp), F.__getattr__(comp))
        elif type(G).__name__ == 'mesh':
            F[:] = _prolong(G, F)
        else:
            raise TransferError('Wrong data type for prolongation, got %s' % type(G))
        return F
