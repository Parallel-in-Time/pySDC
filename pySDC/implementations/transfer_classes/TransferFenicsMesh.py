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

        P is collected column by column from df.interpolate of the coarse basis functions.
        dolfin 2019.1.0's PETScDMCollection.create_transfer_matrix segfaults, and going through
        scipy has the side benefit that P^T is then free.

        Costs one df.interpolate per coarse dof, so it is fine for moderate coarse spaces and
        wants the analytic construction (fine dof coords -> coarse cell -> coarse basis) if the
        coarse level ever gets large.
        """
        if self._Pmat is None:
            Vc, Vf = self.coarse_prob.init, self.fine_prob.init
            e_c, out_f = df.Function(Vc), df.Function(Vf)
            rows, cols, vals = [], [], []
            for j in range(Vc.dim()):
                e_c.vector().zero()
                e_c.vector()[j] = 1.0
                out_f.assign(df.interpolate(e_c, Vf))
                col = out_f.vector()[:]
                nz = np.nonzero(np.abs(col) > 1e-13)[0]
                rows.extend(nz)
                cols.extend([j] * len(nz))
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
