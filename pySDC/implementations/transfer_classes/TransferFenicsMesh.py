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
        self._Pmat = None
        self._l2 = None

    @property
    def Pmat(self):
        """
        Prolongation matrix P (coarse -> fine), assembled once on first use.

        P is the inclusion of the coarse space in the fine one: column j holds the coarse basis
        function phi_j expanded in the fine basis. It is built cell by cell -- for every fine cell
        the enclosing coarse cell is found from the fine cell's midpoint, which is never on a coarse
        facet, and the coarse basis is evaluated in *that* cell.

        Going through df.interpolate instead, as this used to, is wrong for discontinuous spaces. A
        fine dof sitting on a coarse facet has two coarse values there, and dolfin's cross-mesh
        interpolate takes whichever cell the bounding-box tree returns first -- silently continuising
        the coarse function. The error is O(1) in the size of the jump and invisible for smooth data,
        but it deletes exactly the part of the coarse correction a DG hierarchy exists to carry. For
        continuous spaces the two constructions agree to machine precision.

        Costs one basis evaluation per fine dof. dolfin 2019.1.0's
        PETScDMCollection.create_transfer_matrix segfaults, and going through scipy has the side
        benefit that P^T is then free.
        """
        if self._Pmat is None:
            Vc, Vf = self.coarse_prob.init, self.fine_prob.init
            tree = Vc.mesh().bounding_box_tree()
            element, dofmap_c = Vc.element(), Vc.dofmap()
            element_f = Vf.element()

            # which component of a mixed space each fine dof belongs to; scalar spaces are all zero
            ncomp = max(Vf.num_sub_spaces(), 1)
            component = np.zeros(Vf.dim(), dtype=int)
            for k in range(Vf.num_sub_spaces()):
                component[Vf.sub(k).dofmap().dofs()] = k

            rows, cols, vals = [], [], []
            seen = set()
            for cell_f in df.cells(Vf.mesh()):
                cell_c = df.Cell(Vc.mesh(), tree.compute_first_entity_collision(cell_f.midpoint()))
                coords, orientation = cell_c.get_vertex_coordinates(), cell_c.orientation()
                dofs_c = dofmap_c.cell_dofs(cell_c.index())
                # per cell, not from tabulate_dof_coordinates: that reports one point per master
                # dof, which on a periodic space is the wrong side of the domain for half the cells
                # and evaluates the coarse basis outside the cell
                x_local = element_f.tabulate_dof_coordinates(cell_f)
                for k, dof_f in enumerate(Vf.dofmap().cell_dofs(cell_f.index())):
                    # a continuous space shares dofs between cells; the second visit is redundant
                    if dof_f in seen:
                        continue
                    seen.add(dof_f)
                    basis = np.asarray(element.evaluate_basis_all(x_local[k], coords, orientation))
                    col = basis.reshape(-1, ncomp)[:, component[dof_f]]
                    nz = np.nonzero(np.abs(col) > 1e-13)[0]
                    rows.extend([dof_f] * len(nz))
                    cols.extend(dofs_c[nz])
                    vals.extend(col[nz])
            self._Pmat = sp.csr_matrix((vals, (rows, cols)), shape=(Vf.dim(), Vc.dim()))
        return self._Pmat

    def restrict_dual(self, F):
        """
        Variational restriction P^T, for quantities living in the dual space.

        The FAS tau of the mass formulation is a load vector, not a nodal function, so it has to
        be restricted with P^T rather than by interpolation. For nested Lagrange spaces
        phi_i^coarse = sum_j P_ji phi_j^fine, which makes P^T exact: it reproduces the coarse
        load vector of the same functional. Interpolating instead reads the load vector as if it
        were a function and is wrong by roughly 2^dim.

        Args:
            F: the fine level data
        """
        PT = self.Pmat.T
        if isinstance(F, fenics_mesh):
            u_coarse = fenics_mesh(self.coarse_prob.init)
            u_coarse.values.vector()[:] = PT.dot(F.values.vector()[:])
        elif isinstance(F, rhs_fenics_mesh):
            u_coarse = rhs_fenics_mesh(self.coarse_prob.init)
            u_coarse.impl.values.vector()[:] = PT.dot(F.impl.values.vector()[:])
            u_coarse.expl.values.vector()[:] = PT.dot(F.expl.values.vector()[:])
        else:
            raise TransferError('Unknown type of fine data, got %s' % type(F))

        return u_coarse

    @property
    def l2_pieces(self):
        """
        Prefactorised coarse mass matrix, :math:`P^T`, and the fine mass matrix, assembled once.

        Factorised up front because it is solved against once per node per sweep; solving from
        scratch each time costs about seven times more.
        """
        if self._l2 is None:

            def mass(V):
                # straight out of PETSc's CSR; `.array()` would densify, and these are ~0.5% dense
                u, v = df.TrialFunction(V), df.TestFunction(V)
                mat = df.as_backend_type(df.assemble(df.inner(u, v) * df.dx)).mat()
                indptr, indices, data = mat.getValuesCSR()
                return sp.csr_matrix((data, indices, indptr), shape=mat.getSize())

            self._l2 = (spla.factorized(mass(self.coarse_prob.init).tocsc()), self.Pmat.T, mass(self.fine_prob.init))
        return self._l2

    def _project_one(self, values):
        """L2-project one fine ``dolfin.Function`` onto the coarse space."""
        solve_coarse, P_transpose, mass_fine = self.l2_pieces
        coarse = df.Function(self.coarse_prob.init)
        coarse.vector()[:] = solve_coarse(P_transpose @ (mass_fine @ values.vector()[:]))
        return coarse

    def project(self, F):
        """
        Restriction of a SOLUTION, by L2 projection: :math:`M_c^{-1} P^T M_f`.

        Point sampling would also do: :math:`R_u` cancels out of the linear FAS iteration, and every
        MLSDC count here is identical either way. It does not cancel across step boundaries, where
        the restricted state seeds the next block, so PFASST is sensitive to it -- on ``grayscott``
        and on the 2d vortex, by up to a factor of two in iterations, and on the vortex at 8 steps by
        an O(1) error in the answer. Sampling is also not well defined on a DG space, where most
        coarse dof points sit on a fine facet. See ``projects/FEM_with_FEniCS`` for the numbers.

        Costs one coarse mass solve, prefactorised in :attr:`l2_pieces`, which is cheaper than the
        cross-mesh ``df.interpolate`` that ``restrict`` uses. The dual quantities go through
        :meth:`restrict_dual` instead.

        Args:
            F: the fine level data
        """
        if isinstance(F, fenics_mesh):
            return fenics_mesh(self._project_one(F.values))
        elif isinstance(F, rhs_fenics_mesh):
            u_coarse = rhs_fenics_mesh(self.coarse_prob.init)
            u_coarse.impl.values = self._project_one(F.impl.values)
            u_coarse.expl.values = self._project_one(F.expl.values)
            return u_coarse
        raise TransferError('Unknown type of fine data, got %s' % type(F))

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
        Prolongation implementation, the exact inclusion via P.

        Not df.interpolate: see the Pmat docstring for why that silently continuises a
        discontinuous coarse function.

        Args:
            G: the coarse level data
        """
        P = self.Pmat
        if isinstance(G, fenics_mesh):
            u_fine = fenics_mesh(self.fine_prob.init)
            u_fine.values.vector()[:] = P.dot(G.values.vector()[:])
        elif isinstance(G, rhs_fenics_mesh):
            u_fine = rhs_fenics_mesh(self.fine_prob.init)
            u_fine.impl.values.vector()[:] = P.dot(G.impl.values.vector()[:])
            u_fine.expl.values.vector()[:] = P.dot(G.expl.values.vector()[:])
        else:
            raise TransferError('Unknown type of coarse data, got %s' % type(G))

        return u_fine
