import dolfin as df
import numpy as np
import scipy.sparse as sp

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
            x_f = Vf.tabulate_dof_coordinates().reshape(Vf.dim(), -1)

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
                for dof_f in Vf.dofmap().cell_dofs(cell_f.index()):
                    # a continuous space shares dofs between cells; the second visit is redundant
                    if dof_f in seen:
                        continue
                    seen.add(dof_f)
                    basis = np.asarray(element.evaluate_basis_all(x_f[dof_f], coords, orientation))
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

    def project(self, F):
        """
        Restriction of a SOLUTION, by interpolation. Deliberately not an L2 projection.

        In FAS the solution restriction cancels out of the linear iteration exactly:

            tau        = C_G(R_u u_F) - R_tau C_F(u_F)
            u_G        = R_u u_F + A_G^-1 R_tau r_F
            correction = P (u_G - R_u u_F) = P A_G^-1 R_tau r_F

        so only R_tau has to be the variational operator (restrict_dual, P^T, a matvec). R_u only
        has to be a sensible solution restriction, so that f_G(R_u u_F) means something once the
        problem is nonlinear. Interpolation qualifies and costs nothing.

        Measured on step_7's heat problem: identical iteration count and error to the exact
        M_c^-1 P^T M_f projection, with zero mass solves (work 67.5 against 97.6). Mass lumping is
        not an alternative here -- high-order Lagrange basis functions are not positive, so the
        mass row sums can vanish, and it failed to converge at all.

        For DG this samples a discontinuous function at coarse dof points, most of which sit on a
        fine facet where it has two values -- 9 of 17 for DG4 under bisection, all of them for DG1
        and DG2. Point sampling is not merely mis-implemented there, it is not a well-defined
        operator. This is left as it is on purpose, and the reason is worth keeping straight,
        because the same ambiguity in prolong() was fatal:

        R_u acts on the SOLUTION, whose jumps are the DG discretisation error, O(h^(p+1)). Measured
        on the burgers example: against the exact L2 projection M_c^-1 P^T M_f, sampling differs by
        4.4 on a random (genuinely jumpy) state of size 3.9, and by 9e-12 on the states the solver
        actually visits. Swapping in the L2 projection changes not one iteration count, at nu = 0.02
        or at nu = 0.002 where the front is ten times steeper. P acts on the coarse CORRECTION, whose
        jumps are O(1) relative to itself -- which is why continuising it deleted the coarse level.

        So this only bites on a solution whose own jumps are O(1): an under-resolved shock, or a
        limited state. If you get there, the exact operator is M_c^-1 P^T M_f, one coarse mass solve,
        block-diagonal for DG.

        Args:
            F: the fine level data
        """
        return self.restrict(F)

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
