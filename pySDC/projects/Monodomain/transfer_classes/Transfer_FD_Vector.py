import numpy as np
import scipy.sparse as sp

import pySDC.helpers.transfer_helper as th
from pySDC.core.Errors import TransferError
from pySDC.core.SpaceTransfer import space_transfer
from pySDC.projects.Monodomain.datatype_classes.FD_Vector import FD_Vector
from scipy.interpolate import BarycentricInterpolator
import scipy.sparse as sprs

# Largely taken from mesh_to_mesh


class FD_to_FD(space_transfer):
    def __init__(self, fine_prob, coarse_prob, params):
        """
        Initialization routine

        Args:
            fine_prob: fine problem
            coarse_prob: coarse problem
            params: parameters for the transfer operators
        """

        # assumes problem has either periodic or neumann-zero boundary conditions

        # invoke super initialization
        super(FD_to_FD, self).__init__(fine_prob, coarse_prob, params)

        if self.params.rorder % 2 != 0:
            raise TransferError('Need even order for restriction')

        if self.params.iorder % 2 != 0:
            raise TransferError('Need even order for interpolation')

        fine_grid = self.fine_prob.parabolic.grids
        coarse_grid = self.coarse_prob.parabolic.grids
        ndim = len(fine_grid)

        # we have a 1d problem
        if ndim == 1:
            fine_grid = fine_grid[0]
            coarse_grid = coarse_grid[0]
            # if number of variables is the same on both levels, Rspace and Pspace are identity
            if fine_grid.size == coarse_grid.size:
                self.Rspace = sp.eye(coarse_grid.size)
                self.Pspace = sp.eye(fine_grid.size)
                self.Ispace = sp.eye(fine_grid.size)
            else:
                self.Pspace = self.interpolation_matrix_1d(fine_grid, coarse_grid, k=self.params.iorder)
                self.Rspace = self.restriction_matrix_1d(fine_grid, coarse_grid, k=self.params.rorder)
                self.Ispace = self.injection_matrix_1d(fine_grid, coarse_grid)

        # we have an n-d problem
        else:
            Rspace = []
            Pspace = []
            Ispace = []
            for i in range(ndim):
                if fine_grid[i].size == coarse_grid[i].size:
                    Rspace.append(sp.eye(fine_grid[i].size))
                    Pspace.append(sp.eye(fine_grid[i].size))
                    Ispace.append(sp.eye(fine_grid[i].size))
                else:
                    Pspace.append(self.interpolation_matrix_1d(fine_grid[i].flatten(), coarse_grid[i].flatten(), k=self.params.iorder))
                    Rspace.append(self.restriction_matrix_1d(fine_grid[i].flatten(), coarse_grid[i].flatten(), k=self.params.rorder))
                    Ispace.append(self.injection_matrix_1d(fine_grid[i].flatten(), coarse_grid[i].flatten()))

            # kronecker 1-d operators for n-d
            self.Pspace = Pspace[0]
            for i in range(1, len(Pspace)):
                self.Pspace = sp.kron(Pspace[i], self.Pspace, format='csc')

            self.Rspace = Rspace[0]
            for i in range(1, len(Rspace)):
                self.Rspace = sp.kron(Rspace[i], self.Rspace, format='csc')

            self.Ispace = Ispace[0]
            for i in range(1, len(Ispace)):
                self.Ispace = sp.kron(Ispace[i], self.Ispace, format='csc')

    def interpolation_matrix_1d(self, fine_grid, coarse_grid, k=2):
        n_f = fine_grid.size
        pad = k // 2
        M = np.zeros((fine_grid.size, coarse_grid.size + 2 * pad))
        padded_c_grid = th.border_padding(coarse_grid, pad, pad)

        for i, p in zip(range(n_f), fine_grid):
            nn = th.next_neighbors(p, padded_c_grid, k)
            # construct the lagrange polynomials for the k neighbors
            circulating_one = np.asarray([1.0] + [0.0] * (k - 1))
            bary_pol = []
            for l in range(k):
                bary_pol.append(BarycentricInterpolator(padded_c_grid[nn], np.roll(circulating_one, l)))
            with np.errstate(divide='ignore'):
                M[i, nn] = np.asarray(list(map(lambda x: x(p), bary_pol)))

        if pad > 0:
            if self.params.periodic:
                for i in range(pad):
                    M[:, pad + i] += M[:, -pad + i]
                    M[:, -pad - 2 - i + 1] += M[:, pad - 1 - i]
            else:
                for i in range(pad):
                    M[:, pad + 1 + i] += M[:, pad - 1 - i]
                    M[:, -pad - 2 - i] += M[:, -pad + i]
            M = M[:, pad:-pad]

        return sprs.csc_matrix(M)

    def restriction_matrix_1d(self, fine_grid, coarse_grid, k=2):
        Rspace = th.interpolation_matrix_1d(fine_grid, coarse_grid, k=k, periodic=False, pad=k // 2, equidist_nested=False).T
        ratio_grids = (coarse_grid[1] - coarse_grid[0]) / (fine_grid[1] - fine_grid[0])
        if k > 0:
            Rspace *= 1 / ratio_grids
        ratio_grids = np.round(ratio_grids).astype(int)
        stencil = Rspace[k // 2, 1 : ratio_grids * k].todense()
        if self.params.periodic:
            for i in range(1, k // 2 + 1):
                Rspace[k // 2 - i, : 2 * k - 2 * i] = stencil[0, 2 * i - 1 :]
                Rspace[k // 2 - i, -(2 * i - 1) :] = stencil[0, : 2 * i - 1]
                if i > 1:
                    Rspace[-1 - (k // 2 - i), -(2 * k - 2 * i) - 1 :] = stencil[0, : 2 * (1 - i)]
                    Rspace[-1 - (k // 2 - i), : 2 * (i - 1)] = stencil[0, 2 * (1 - i) :]
        else:
            for i in range(1, k // 2 + 1):
                Rspace[k // 2 - i, : 2 * k - 2 * i] = stencil[0, 2 * i - 1 :]
                Rspace[k // 2 - i, 1 : 2 * i] += np.flip(stencil[0, : 2 * i - 1])
                Rspace[-1 - (k // 2 - i), -(2 * k - 2 * i) :] = stencil[0, : -(2 * i - 1)]
                Rspace[-1 - (k // 2 - i), -2 * i : -1] += np.flip(stencil[0, 1 - 2 * i :])

        return Rspace

    def injection_matrix_1d(self, fine_grid, coarse_grid):
        Ispace = th.interpolation_matrix_1d(coarse_grid, fine_grid, k=1, periodic=False, pad=0, equidist_nested=False)
        return Ispace

    def restrict(self, F):
        """
        Restriction implementation
        Args:
            F: the fine level data (easier to access than via the fine attribute)
        """
        G = FD_Vector(self.coarse_prob.init)
        G.values[:] = self.Rspace @ F.values
        return G

    def prolong(self, G):
        """
        Prolongation implementation
        Args:
            G: the coarse level data (easier to access than via the coarse attribute)
        """
        F = FD_Vector(self.fine_prob.init)
        F.values[:] = self.Pspace @ G.values
        return F

    def inject(self, F):
        """
        Injection implementation
        Args:
            F: the fine level data (easier to access than via the fine attribute)
        """
        G = FD_Vector(self.coarse_prob.init)
        G.values[:] = self.Ispace @ F.values
        return G
