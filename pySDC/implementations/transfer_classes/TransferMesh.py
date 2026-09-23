import numpy as np
import scipy.sparse as sp

import pySDC.helpers.transfer_helper as th
from pySDC.core.errors import TransferError
from pySDC.core.space_transfer import SpaceTransfer


def _unit_grid(nvars, periodic):
    """
    Grid of one spatial dimension, scaled to a domain of length one.

    The interpolation weights depend only on ratios of distances, so building the grids on the unit
    domain instead of on the problem's own ``dx`` gives the same operator for any domain length.
    For periodic grids it is also what makes the operator correct at all: the helpers in
    ``transfer_helper`` hardcode a period of one.

    Args:
        nvars (int): number of degrees of freedom in this dimension
        periodic (bool): whether this dimension is periodic

    Returns:
        np.ndarray: the grid, within [0, 1)
    """
    return np.arange(nvars) / nvars if periodic else (np.arange(nvars) + 1) / (nvars + 1)


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
                fine_grid = _unit_grid(self.fine_prob.nvars, self.params.periodic)
                coarse_grid = _unit_grid(self.coarse_prob.nvars, self.params.periodic)

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
                    fine_grid = _unit_grid(self.fine_prob.nvars[i], self.params.periodic)
                    coarse_grid = _unit_grid(self.coarse_prob.nvars[i], self.params.periodic)

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

        # Carry the operators at the precision of what they produce, rather than always at float64.
        # A float64 operator applied to a reduced-precision vector upcasts, so the transfer would do
        # its work in double and only round on the way into the destination, which is the one place
        # a reduced-precision level would silently keep paying full freight. `promote_types` with
        # float32 keeps it legal for SciPy, which has no half-precision sparse matrix.
        self.Rspace = self.Rspace.astype(np.promote_types(self.coarse_prob.init[-1], np.float32))
        self.Pspace = self.Pspace.astype(np.promote_types(self.fine_prob.init[-1], np.float32))

        # Which side of the PCI bus this runs on is a property of the problem, not something the
        # transfer is told -- the same way `TransferMesh_MPIFFT` decides it. The operators are
        # assembled with SciPy either way, since that work is small, one-off and full of host-side
        # index arithmetic; only the finished matrices move.
        if 'cupy' in self.fine_prob.dtype_u.__name__.lower():
            import cupyx.scipy.sparse as csp

            self.Rspace = csp.csr_matrix(self.Rspace)
            self.Pspace = csp.csr_matrix(self.Pspace)

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
                _restrict(getattr(F, comp), getattr(G, comp))
        elif type(F).__name__ in ['mesh', 'cupy_mesh']:
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
                _prolong(getattr(G, comp), getattr(F, comp))
        elif type(G).__name__ in ['mesh', 'cupy_mesh']:
            F[:] = _prolong(G, F)
        else:
            raise TransferError('Wrong data type for prolongation, got %s' % type(G))
        return F
