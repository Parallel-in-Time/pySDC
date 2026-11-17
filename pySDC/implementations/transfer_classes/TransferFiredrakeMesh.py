from firedrake import assemble, interpolate, TestFunction, TransferManager

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


def run_transfer(op, source, target, buffers):
    """
    Run the transfer `op` (``'prolong'``, ``'restrict'`` or ``'inject'``) of Firedrake's transfer manager from `source`
    to `target`, through buffers kept in `buffers`.

    The manager records the data versions of every pair of source and target it has seen, and never forgets a pair, so
    transferring fresh objects every time adds one record per call, which a long run accumulates. Through the buffers
    it sees the same pair every time. `buffers` is a dict owned by the caller, which keeps the buffers as long as it
    lives.

    Args:
        op (str): The transfer
        source (firedrake.Function or firedrake.Cofunction): What to transfer
        target (firedrake.Function or firedrake.Cofunction): Where the result goes
        buffers (dict): The caller's buffers, filled on first use

    Returns:
        `target`
    """
    key = (op, source.function_space(), target.function_space())
    if key not in buffers:
        buffers[key] = (type(source)(source.function_space()), type(target)(target.function_space()))
    source_buffer, target_buffer = buffers[key]
    source_buffer.assign(source)
    getattr(_transfer_manager, op)(source_buffer, target_buffer)
    return target.assign(target_buffer)


class MeshToMeshFiredrake(SpaceTransfer):
    """
    This implementation can restrict and prolong between Firedrake meshes, by interpolation.

    On the same mesh, between Lagrange spaces of different degrees (p-coarsening), interpolation into the finer space
    is the exact inclusion of the coarse space. For the mass-matrix formulation, used through
    :class:`~pySDC.implementations.transfer_classes.BaseTransfer_mass.base_transfer_mass`, :meth:`restrict_dual`
    restricts load vectors with the transpose of that prolongation and :meth:`project` restricts solutions by
    :math:`L^2` projection, which are then exact as well. For meshes of a hierarchy, use
    :class:`MeshToMeshFiredrakeHierarchy`.
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

    def restrict_dual(self, F):
        r"""
        Restrict a load vector (dual data) with :math:`P^T`, the adjoint of the interpolation :meth:`prolong` uses.

        For nested spaces, such as lower-degree Lagrange elements on the same mesh, every coarse basis function is a
        combination of fine ones through :math:`P`, so this gives exactly the coarse load vector of the same
        functional.

        Args:
            F (firedrake_mesh): Fine dual data, e.g. the FAS correction or :math:`M u_0`

        Returns:
            firedrake_dual_mesh: Coarse dual data
        """
        if not (isinstance(F, firedrake_mesh) and F.is_dual):
            raise TransferError(f'restrict_dual needs dual data, got {type(F)}')
        G = firedrake_dual_mesh(self.coarse_prob.init)
        assemble(interpolate(TestFunction(self.coarse_prob.init), F.functionspace), tensor=G.functionspace)
        return G

    def project(self, F):
        r"""
        Restrict a solution by :math:`L^2` projection, :math:`M_c^{-1} P^T M_f u`, with :math:`P^T` as in
        :meth:`restrict_dual`. The mass matrices are assembled once per space.

        Args:
            F (firedrake_mesh): Fine solution

        Returns:
            firedrake_mesh: Coarse solution
        """
        if not (isinstance(F, firedrake_mesh) and not F.is_dual):
            raise TransferError(f'project needs primal data, got {type(F)}')
        load = firedrake_mesh(riesz_map(self.fine_prob.init)(F.functionspace))
        return firedrake_mesh(riesz_map(self.coarse_prob.init)(self.restrict_dual(load).functionspace))


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

    def __init__(self, *args, **kwargs):
        """Initialization, see :class:`~pySDC.core.space_transfer.SpaceTransfer`."""
        super().__init__(*args, **kwargs)
        #: the buffers of :func:`run_transfer`, one pair per kind of transfer
        self.buffers = {}

    def _restrict(self, u_fine, u_coarse):
        """Perform restriction in Firedrake"""
        run_transfer('inject', u_fine.functionspace, u_coarse.functionspace, self.buffers)

    def _prolong(self, u_coarse, u_fine):
        """Perform prolongation in Firedrake"""
        run_transfer('prolong', u_coarse.functionspace, u_fine.functionspace, self.buffers)

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
            # primal like G: the fine problem's own right-hand sides may be load vectors
            F = IMEX_firedrake_mesh(self.fine_prob.init)
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
        run_transfer('restrict', F.functionspace, G.functionspace, self.buffers)
        return G

    def project(self, F):
        r"""
        Restrict a solution by :math:`L^2` projection, :math:`M_c^{-1} P^T M_f u`.

        Unlike injection, this is well defined for discontinuous spaces and does not depend on point values. The mass
        matrices are assembled once per space, see
        :func:`~pySDC.implementations.datatype_classes.firedrake_mesh.riesz_map`.

        Args:
            F (firedrake_mesh): Fine solution

        Returns:
            firedrake_mesh: Coarse solution
        """
        if not (isinstance(F, firedrake_mesh) and not F.is_dual):
            raise TransferError(f'project needs primal data, got {type(F)}')
        coarse_load = Cofunction(self.coarse_prob.init.dual())
        run_transfer('restrict', riesz_map(self.fine_prob.init)(F.functionspace), coarse_load, self.buffers)
        return firedrake_mesh(riesz_map(self.coarse_prob.init)(coarse_load))
