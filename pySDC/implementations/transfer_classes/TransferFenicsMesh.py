import dolfin as df
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from pySDC.core.errors import TransferError
from pySDC.core.space_transfer import SpaceTransfer
from pySDC.implementations.datatype_classes.fenics_mesh import fenics_mesh, rhs_fenics_mesh


class mesh_to_mesh_fenics(SpaceTransfer):
    """
    This implementation can restrict and prolong between fenics meshes
    """

    def __init__(self, fine_prob, coarse_prob, params):
        super().__init__(fine_prob, coarse_prob, params)
        self._l2 = None

    @property
    def l2_pieces(self):
        r"""
        The pieces of the L2 projection onto the coarse space, assembled once.

        The coarse space is a subspace of the fine one, so writing :math:`P` for the coarse basis
        expressed in the fine space, the projection is :math:`M_c^{-1} P^T M_f`. The coarse mass
        matrix is factorised up front because it is solved against once per node per sweep.

        :math:`P` is collected column by column, one ``df.interpolate`` per coarse dof. That is
        exact for the continuous spaces this class is used with, where a coarse basis function is
        continuous and so its point values determine it. It is *not* exact for a discontinuous
        space, where a fine dof on a coarse facet has two coarse values and ``df.interpolate``
        returns whichever the bounding-box tree finds first.
        """
        if self._l2 is None:
            Vc, Vf = self.coarse_prob.init, self.fine_prob.init

            def mass(V):
                u, v = df.TrialFunction(V), df.TestFunction(V)
                mat = df.as_backend_type(df.assemble(df.inner(u, v) * df.dx)).mat()
                indptr, indices, data = mat.getValuesCSR()
                return sp.csr_matrix((data, indices, indptr), shape=mat.getSize())

            P = np.zeros((Vf.dim(), Vc.dim()))
            basis = df.Function(Vc)
            for j in range(Vc.dim()):
                basis.vector().zero()
                basis.vector()[j] = 1.0
                P[:, j] = df.interpolate(basis, Vf).vector()[:]

            self._l2 = (spla.factorized(mass(Vc).tocsc()), sp.csr_matrix(P).T, mass(Vf))
        return self._l2

    def _project(self, values):
        """L2-project one fine ``dolfin.Function`` onto the coarse space."""
        solve_coarse, P_transpose, mass_fine = self.l2_pieces
        coarse = df.Function(self.coarse_prob.init)
        coarse.vector()[:] = solve_coarse(P_transpose @ (mass_fine @ values.vector()[:]))
        return coarse

    def project(self, F):
        r"""
        Restriction implementation via projection.

        Not ``df.project``, which builds a form over the coarse mesh out of a function living on the
        fine one. Dolfin does not support that and does not complain: when the two levels share an
        element degree it returns the nodal interpolant rather than a projection, and when they do
        not it returns an inexact one. See :attr:`l2_pieces`.

        Args:
            F: the fine level data
        """
        if isinstance(F, fenics_mesh):
            u_coarse = fenics_mesh(self._project(F.values))
        elif isinstance(F, rhs_fenics_mesh):
            u_coarse = rhs_fenics_mesh(self.coarse_prob.init)
            u_coarse.impl.values = self._project(F.impl.values)
            u_coarse.expl.values = self._project(F.expl.values)
        else:
            raise TransferError('Unknown type of fine data, got %s' % type(F))

        return u_coarse

    def restrict(self, F):
        """
        Restriction implementation

        Args:
            F: the fine level data
        """
        if isinstance(F, fenics_mesh):
            u_coarse = fenics_mesh(df.interpolate(F.values, self.coarse_prob.init))
        elif isinstance(F, rhs_fenics_mesh):
            u_coarse = rhs_fenics_mesh(self.coarse_prob.init)
            u_coarse.impl.values = df.interpolate(F.impl.values, self.coarse_prob.init)
            u_coarse.expl.values = df.interpolate(F.expl.values, self.coarse_prob.init)
        else:
            raise TransferError('Unknown type of fine data, got %s' % type(F))

        return u_coarse

    def prolong(self, G):
        """
        Prolongation implementation

        Args:
            G: the coarse level data
        """
        if isinstance(G, fenics_mesh):
            u_fine = fenics_mesh(df.interpolate(G.values, self.fine_prob.init))
        elif isinstance(G, rhs_fenics_mesh):
            u_fine = rhs_fenics_mesh(self.fine_prob.init)
            u_fine.impl.values = df.interpolate(G.impl.values, self.fine_prob.init)
            u_fine.expl.values = df.interpolate(G.expl.values, self.fine_prob.init)
        else:
            raise TransferError('Unknown type of coarse data, got %s' % type(G))

        return u_fine
