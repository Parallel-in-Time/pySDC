
import cupy as cp
import cupyx.scipy.sparse as csp

import pySDC.helpers.transfer_helper_gpu as th
from pySDC.core.Errors import TransferError
from pySDC.core.SpaceTransfer import space_transfer
from pySDC.implementations.datatype_classes.cupy_mesh import cupy_mesh, imex_cupy_mesh, comp2_cupy_mesh


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
                self.Rspace = csp.eye(self.coarse_prob.params.nvars)
                self.Pspace = csp.eye(self.fine_prob.params.nvars)
            # assemble restriction as transpose of interpolation
            else:

                if not self.params.periodic:
                    fine_grid = cp.array([(i + 1) * self.fine_prob.dx for i in range(self.fine_prob.params.nvars)])
                    coarse_grid = cp.array(
                        [(i + 1) * self.coarse_prob.dx for i in range(self.coarse_prob.params.nvars)])
                else:
                    fine_grid = cp.array([i * self.fine_prob.dx for i in range(self.fine_prob.params.nvars)])
                    coarse_grid = cp.array([i * self.coarse_prob.dx for i in range(self.coarse_prob.params.nvars)])

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
                    Rspace.append(csp.eye(self.coarse_prob.params.nvars[i]))
                    Pspace.append(csp.eye(self.fine_prob.params.nvars[i]))
                # assemble restriction as transpose of interpolation
                else:

                    if not self.params.periodic:
                        fine_grid = cp.array(
                            [(j + 1) * self.fine_prob.dx for j in range(self.fine_prob.params.nvars[i])])
                        coarse_grid = cp.array(
                            [(j + 1) * self.coarse_prob.dx for j in range(self.coarse_prob.params.nvars[i])])
                    else:
                        fine_grid = cp.array([j * self.fine_prob.dx for j in range(self.fine_prob.params.nvars[i])])
                        coarse_grid = cp.array(
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
                        mat = th.interpolation_matrix_1d(fine_grid, coarse_grid,
                                                         k=self.params.rorder,
                                                         periodic=self.params.periodic,
                                                         equidist_nested=self.params.equidist_nested).T
                        Rspace.append(restr_factor * mat)

            # kronecker 1-d operators for n-d
            self.Pspace = Pspace[0]
            for i in range(1, len(Pspace)):
                self.Pspace = csp.kron(self.Pspace, Pspace[i], format='csc')

            self.Rspace = Rspace[0]
            for i in range(1, len(Rspace)):
                self.Rspace = csp.kron(self.Rspace, Rspace[i], format='csc')

    def restrict(self, F):
        """
        Restriction implementation
        Args:
            F: the fine level data (easier to access than via the fine attribute)
        """
        if isinstance(F, cupy_mesh):
            G = self.coarse_prob.dtype_u(self.coarse_prob.init)
            if hasattr(self.fine_prob, 'ncomp'):
                for i in range(self.fine_prob.ncomp):
                    tmpF = F[..., i].flatten()
                    tmpG = self.Rspace.dot(tmpF)
                    G[..., i] = tmpG.reshape(self.coarse_prob.params.nvars)
            else:
                tmpF = F.flatten()
                tmpG = self.Rspace.dot(tmpF)
                G[:] = tmpG.reshape(self.coarse_prob.params.nvars)
        elif isinstance(F, imex_cupy_mesh):
            G = self.coarse_prob.dtype_f(self.coarse_prob.init)
            if hasattr(self.fine_prob, 'ncomp'):
                for i in range(self.fine_prob.ncomp):
                    tmpF = F.impl[..., i].flatten()
                    tmpG = self.Rspace.dot(tmpF)
                    G.impl[..., i] = tmpG.reshape(self.coarse_prob.params.nvars)
                    tmpF = F.expl[..., i].flatten()
                    tmpG = self.Rspace.dot(tmpF)
                    G.expl[..., i] = tmpG.reshape(self.coarse_prob.params.nvars)
            else:
                tmpF = F.impl.flatten()
                tmpG = self.Rspace.dot(tmpF)
                G.impl[:] = tmpG.reshape(self.coarse_prob.params.nvars)
                tmpF = F.expl.flatten()
                tmpG = self.Rspace.dot(tmpF)
                G.expl[:] = tmpG.reshape(self.coarse_prob.params.nvars)
        elif isinstance(F, comp2_cupy_mesh):
            G = self.coarse_prob.dtype_f(self.coarse_prob.init)
            if hasattr(self.fine_prob, 'ncomp'):
                for i in range(self.fine_prob.ncomp):
                    tmpF = F.comp1[..., i].flatten()
                    tmpG = self.Rspace.dot(tmpF)
                    G.comp1[..., i] = tmpG.reshape(self.coarse_prob.params.nvars)
                    tmpF = F.comp2[..., i].flatten()
                    tmpG = self.Rspace.dot(tmpF)
                    G.comp2[..., i] = tmpG.reshape(self.coarse_prob.params.nvars)
            else:
                tmpF = F.comp1.flatten()
                tmpG = self.Rspace.dot(tmpF)
                G.comp1[:] = tmpG.reshape(self.coarse_prob.params.nvars)
                tmpF = F.comp2.flatten()
                tmpG = self.Rspace.dot(tmpF)
                G.comp2[:] = tmpG.reshape(self.coarse_prob.params.nvars)
        else:
            raise TransferError('Wrong data type for restriction, got %s' % type(F))
        return G

    def prolong(self, G):
        """
        Prolongation implementation
        Args:
            G: the coarse level data (easier to access than via the coarse attribute)
        """
        if isinstance(G, cupy_mesh):
            F = self.fine_prob.dtype_u(self.fine_prob.init)
            if hasattr(self.fine_prob, 'ncomp'):
                for i in range(self.fine_prob.ncomp):
                    tmpG = G[..., i].flatten()
                    tmpF = self.Pspace.dot(tmpG)
                    F[..., i] = tmpF.reshape(self.fine_prob.params.nvars)
            else:
                tmpG = G.flatten()
                tmpF = self.Pspace.dot(tmpG)
                F[:] = tmpF.reshape(self.fine_prob.params.nvars)
        elif isinstance(G, imex_cupy_mesh):
            F = self.fine_prob.dtype_f(self.fine_prob.init)
            if hasattr(self.fine_prob, 'ncomp'):
                for i in range(self.fine_prob.ncomp):
                    tmpG = G.impl[..., i].flatten()
                    tmpF = self.Pspace.dot(tmpG)
                    F.impl[..., i] = tmpF.reshape(self.fine_prob.params.nvars)
                    tmpG = G.expl[..., i].flatten()
                    tmpF = self.Rspace.dot(tmpG)
                    F.expl[..., i] = tmpF.reshape(self.fine_prob.params.nvars)
            else:
                tmpG = G.impl.flatten()
                tmpF = self.Pspace.dot(tmpG)
                F.impl[:] = tmpF.reshape(self.fine_prob.params.nvars)
                tmpG = G.expl.flatten()
                tmpF = self.Pspace.dot(tmpG)
                F.expl[:] = tmpF.reshape(self.fine_prob.params.nvars)
        elif isinstance(G, comp2_cupy_mesh):
            F = self.fine_prob.dtype_f(self.fine_prob.init)
            if hasattr(self.fine_prob, 'ncomp'):
                for i in range(self.fine_prob.ncomp):
                    tmpG = G.comp1[..., i].flatten()
                    tmpF = self.Pspace.dot(tmpG)
                    F.comp1[..., i] = tmpF.reshape(self.fine_prob.params.nvars)
                    tmpG = G.comp2[..., i].flatten()
                    tmpF = self.Rspace.dot(tmpG)
                    F.comp2[..., i] = tmpF.reshape(self.fine_prob.params.nvars)
            else:
                tmpG = G.comp1.flatten()
                tmpF = self.Pspace.dot(tmpG)
                F.comp1[:] = tmpF.reshape(self.fine_prob.params.nvars)
                tmpG = G.comp2.flatten()
                tmpF = self.Pspace.dot(tmpG)
                F.comp2[:] = tmpF.reshape(self.fine_prob.params.nvars)
        else:
            raise TransferError('Wrong data type for prolongation, got %s' % type(G))
        return F
