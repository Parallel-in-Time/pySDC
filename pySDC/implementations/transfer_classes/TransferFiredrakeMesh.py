from firedrake import assemble, prolong, inject
from firedrake.__future__ import interpolate

from pySDC.core.errors import TransferError
from pySDC.core.space_transfer import SpaceTransfer
from pySDC.implementations.datatype_classes.firedrake_mesh import firedrake_mesh, IMEX_firedrake_mesh


class MeshToMeshFiredrake(SpaceTransfer):
    """
    This implementation can restrict and prolong between Firedrake meshes
    """

    def restrict(self, F):
        """
        Restrict from fine to coarse grid

        Args:
            F: the fine level data

        Returns:
            Coarse level data
        """
        if isinstance(F, firedrake_mesh):
            u_coarse = self.coarse_prob.dtype_u(assemble(interpolate(F.functionspace, self.coarse_prob.init)))
        elif isinstance(F, IMEX_firedrake_mesh):
            u_coarse = IMEX_firedrake_mesh(self.coarse_prob.init)
            u_coarse.impl.functionspace.assign(assemble(interpolate(F.impl.functionspace, self.coarse_prob.init)))
            u_coarse.expl.functionspace.assign(assemble(interpolate(F.expl.functionspace, self.coarse_prob.init)))
        else:
            raise TransferError('Unknown type of fine data, got %s' % type(F))

        return u_coarse

    def prolong(self, G):
        """
        Prolongate from coarse to fine grid

        Args:
            G: the coarse level data

        Returns:
            fine level data
        """
        if isinstance(G, firedrake_mesh):
            u_fine = self.fine_prob.dtype_u(assemble(interpolate(G.functionspace, self.fine_prob.init)))
        elif isinstance(G, IMEX_firedrake_mesh):
            u_fine = IMEX_firedrake_mesh(self.fine_prob.init)
            u_fine.impl.functionspace.assign(assemble(interpolate(G.impl.functionspace, self.fine_prob.init)))
            u_fine.expl.functionspace.assign(assemble(interpolate(G.expl.functionspace, self.fine_prob.init)))
        else:
            raise TransferError('Unknown type of coarse data, got %s' % type(G))

        return u_fine


class MeshToMeshFiredrakeHierarchy(SpaceTransfer):
    """
    This implementation can restrict and prolong between Firedrake meshes that are generated from a hierarchy.
    Example:

    ```
    from firedrake import *

    mesh = UnitSquareMesh(8, 8)
    hierarchy = MeshHierarchy(mesh, 4)

    mesh = hierarchy[-1]
    ```
    """

    @staticmethod
    def _restrict(u_fine, u_coarse):
        """Perform restriction in Firedrake"""
        inject(u_fine.functionspace, u_coarse.functionspace)

    @staticmethod
    def _prolong(u_coarse, u_fine):
        """Perform prolongation in Firedrake"""
        prolong(u_coarse.functionspace, u_fine.functionspace)

    def restrict(self, F):
        """
        Restrict from fine to coarse grid

        Args:
            F: the fine level data

        Returns:
            Coarse level data
        """
        if isinstance(F, firedrake_mesh):
            G = self.coarse_prob.u_init
            self._restrict(u_fine=F, u_coarse=G)
        elif isinstance(F, IMEX_firedrake_mesh):
            G = IMEX_firedrake_mesh(self.coarse_prob.init)
            self._restrict(u_fine=F.impl, u_coarse=G.impl)
            self._restrict(u_fine=F.expl, u_coarse=G.expl)
        else:
            raise TransferError('Unknown type of fine data, got %s' % type(F))

        return G

    def prolong(self, G):
        """
        Prolongate from coarse to fine grid

        Args:
            G: the coarse level data

        Returns:
            fine level data
        """
        if isinstance(G, firedrake_mesh):
            F = self.fine_prob.u_init
            self._prolong(u_coarse=G, u_fine=F)
        elif isinstance(G, IMEX_firedrake_mesh):
            F = self.fine_prob.f_init
            self._prolong(u_coarse=G.impl, u_fine=F.impl)
            self._prolong(u_coarse=G.expl, u_fine=F.expl)
        else:
            raise TransferError('Unknown type of coarse data, got %s' % type(G))

        return F
