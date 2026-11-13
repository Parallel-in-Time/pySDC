from firedrake import assemble, interpolate, TransferManager

from pySDC.core.errors import TransferError
from pySDC.core.space_transfer import SpaceTransfer
from pySDC.implementations.datatype_classes.firedrake_mesh import (
    firedrake_mesh,
    firedrake_dual_mesh,
    IMEX_firedrake_mesh,
    riesz_map,
)
from firedrake import Cofunction

# Firedrake's plain `inject` and `prolong` reject mixed spaces and give NaN when injecting H(div) spaces such as BDM
# on immersed manifolds (Gusto's sphere); the transfer manager handles both.
_transfer_manager = TransferManager()


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

    For the mass-matrix formulation, used through
    :class:`~pySDC.implementations.transfer_classes.BaseTransfer_mass.base_transfer_mass`, it also restricts load
    vectors with :meth:`restrict_dual` and solutions by :math:`L^2` projection with :meth:`project`.
    Example:

    .. code-block:: python

        from firedrake import *

        mesh = UnitSquareMesh(8, 8)
        hierarchy = MeshHierarchy(mesh, 4)

        mesh = hierarchy[-1]

    """

    @staticmethod
    def _restrict(u_fine, u_coarse):
        """Perform restriction in Firedrake"""
        _transfer_manager.inject(u_fine.functionspace, u_coarse.functionspace)

    @staticmethod
    def _prolong(u_coarse, u_fine):
        """Perform prolongation in Firedrake"""
        _transfer_manager.prolong(u_coarse.functionspace, u_fine.functionspace)

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

    def restrict_dual(self, F):
        r"""
        Restrict a load vector (dual data) with :math:`P^T`, the transpose of the prolongation.

        For nested spaces, every coarse basis function is a combination of fine ones through :math:`P`, so this gives
        exactly the coarse load vector of the same functional. Injecting instead would read the load vector as if it
        were a function.

        Args:
            F (firedrake_mesh): Fine dual data, e.g. the FAS correction or :math:`M u_0`

        Returns:
            firedrake_dual_mesh: Coarse dual data
        """
        if not (isinstance(F, firedrake_mesh) and F.is_dual):
            raise TransferError(f'restrict_dual needs dual data, got {type(F)}')
        G = firedrake_dual_mesh(self.coarse_prob.init)
        _transfer_manager.restrict(F.functionspace, G.functionspace)
        return G

    def project(self, F):
        r"""
        Restrict a solution by :math:`L^2` projection, :math:`M_c^{-1} P^T M_f u`.

        Unlike injection, this is well defined for discontinuous spaces and does not depend on point values. The mass
        matrices are factored once per space, see
        :func:`~pySDC.implementations.datatype_classes.firedrake_mesh.riesz_map`.

        Args:
            F (firedrake_mesh): Fine solution

        Returns:
            firedrake_mesh: Coarse solution
        """
        if not (isinstance(F, firedrake_mesh) and not F.is_dual):
            raise TransferError(f'project needs primal data, got {type(F)}')
        coarse_load = Cofunction(self.coarse_prob.init.dual())
        _transfer_manager.restrict(riesz_map(self.fine_prob.init)(F.functionspace), coarse_load)
        return firedrake_mesh(riesz_map(self.coarse_prob.init)(coarse_load))
