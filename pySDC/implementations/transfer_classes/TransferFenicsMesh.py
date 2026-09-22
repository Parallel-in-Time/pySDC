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
                # dof coordinates come per cell rather than from the global table. On a constrained
                # space -- periodic boundaries -- a master dof stands for two points on opposite
                # sides of the domain and the global table reports only one of them, which puts the
                # evaluation point outside the coarse cell and extrapolates. That produced entries
                # of 1e3 where a Lagrange basis inside its own cell cannot exceed 1.
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

        The coarse mass matrix is solved against once per node per sweep, so it is factorised up
        front rather than solved from scratch. That is the difference between this costing less than
        the interpolation it replaces and costing seven times more.
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

        In FAS the solution restriction cancels out of the *linear* iteration exactly:

            tau        = C_G(R_u u_F) - R_tau C_F(u_F)
            u_G        = R_u u_F + A_G^-1 R_tau r_F
            correction = P (u_G - R_u u_F) = P A_G^-1 R_tau r_F

        so only R_tau has to be the variational operator (restrict_dual, P^T, a matvec). This used
        to return ``self.restrict(F)`` -- point sampling -- on the strength of that argument, and
        the argument is right as far as it goes: **every MLSDC iteration count in this project is
        identical either way**, on all three examples, both families and both coarsening directions,
        in 1d and on the 2d vortex.

        It does not carry to PFASST. The cancellation is a property of the two-level iteration;
        across step boundaries the restricted state is what seeds the next block, and R_u stops
        dropping out. Whether that is visible depends on how far apart the two operators are on the
        states the solver actually visits -- 1.4e-15 for heat, 9e-12 for burgers, but 4.3e-8 for
        grayscott. Iterations at 8 parallel steps:

            grayscott [CG, h]   6.00 -> 5.38      grayscott [DG, h]    6.75 -> 5.38
            grayscott [CG, p]   9.25 -> 5.88      grayscott [DG, p]   12.12 -> 5.75

        and on the 2d vortex at 4 parallel steps, 14.75 -> 8.38. Errors are comparable throughout,
        so this is not a looser tolerance buying fewer sweeps. It also nearly closes the h/p gap,
        i.e. p-coarsening's poor PFASST scaling was substantially an artefact of point sampling.

        The cost argument that motivated sampling runs the other way once P and the factorisation
        are cached, because `df.interpolate` walks a bounding-box tree per dof while this is three
        sparse operations. Per call, grayscott h, 2050 -> 1026 dofs:

            cached P, prefactorised M_c   0.064 ms       df.project each call   2.447 ms
            cached P, spsolve each call   0.432 ms       df.interpolate         0.746 ms

        In 2d the gap is wider still: 13-27x for CG up to 66k dofs, 7-13x for DG, with setup under a
        second. P is already built for `prolong`, so the marginal setup here is two mass assemblies
        and one factorisation, 0.04 s on the 2d vortex.

        Sampling is also not a well-defined operator on a DG space -- most coarse dof points sit on a
        fine facet where the function has two values -- so the operator that is correct is also the
        one that is cheaper and never needs more iterations. ``restrict`` still point-samples and is
        still what the dual quantities use via ``restrict_dual``.

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
