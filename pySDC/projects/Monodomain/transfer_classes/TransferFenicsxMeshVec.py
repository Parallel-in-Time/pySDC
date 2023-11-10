import dolfinx as df
from petsc4py import PETSc
from pySDC.core.Errors import TransferError
from pySDC.core.SpaceTransfer import space_transfer
from pySDC.projects.Monodomain.datatype_classes.fenicsx_mesh_vec import fenicsx_mesh_vec, rhs_fenicsx_mesh_vec, exp_rhs_fenicsx_mesh_vec


class mesh_to_mesh_fenicsx(space_transfer):
    """
    This implementation can restrict and prolong between fenicsx meshes
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
        super(mesh_to_mesh_fenicsx, self).__init__(fine_prob, coarse_prob, params)

        pass

    def restrict(self, F):
        """
        Restriction implementation

        Args:
            F: the fine level data
        """
        if isinstance(F, fenicsx_mesh_vec):
            u_coarse = fenicsx_mesh_vec(init=self.coarse_prob.init, size=F.size)
            F.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
            u_coarse.interpolate(F)
        elif isinstance(F, rhs_fenicsx_mesh_vec):
            u_coarse = rhs_fenicsx_mesh_vec(init=self.coarse_prob.init, size=F.size)
            F.impl.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
            F.expl.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
            u_coarse.impl.interpolate(F.impl)
            u_coarse.expl.interpolate(F.expl)
        elif isinstance(F, exp_rhs_fenicsx_mesh_vec):
            u_coarse = exp_rhs_fenicsx_mesh_vec(init=self.coarse_prob.init, size=F.size)
            F.impl.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
            F.expl.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
            F.exp.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
            u_coarse.impl.interpolate(F.impl)
            u_coarse.expl.interpolate(F.expl)
            u_coarse.exp.interpolate(F.exp)
        else:
            raise TransferError('Unknown type of fine data, got %s' % type(F))

        return u_coarse

    def prolong(self, G):
        """
        Prolongation implementation

        Args:
            G: the coarse level data
        """
        if isinstance(G, fenicsx_mesh_vec):
            u_fine = fenicsx_mesh_vec(init=self.fine_prob.init, size=G.size)
            G.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
            u_fine.interpolate(G)
        elif isinstance(G, rhs_fenicsx_mesh_vec):
            u_fine = rhs_fenicsx_mesh_vec(init=self.fine_prob.init, size=G.size)
            G.impl.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
            G.expl.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
            u_fine.impl.interpolate(G.impl)
            u_fine.expl.interpolate(G.expl)
        elif isinstance(G, exp_rhs_fenicsx_mesh_vec):
            u_fine = exp_rhs_fenicsx_mesh_vec(init=self.fine_prob.init, size=G.size)
            G.impl.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
            G.expl.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
            G.exp.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
            u_fine.impl.interpolate(G.impl)
            u_fine.expl.interpolate(G.expl)
            u_fine.exp.interpolate(G.exp)
        else:
            raise TransferError('Unknown type of coarse data, got %s' % type(G))

        return u_fine
