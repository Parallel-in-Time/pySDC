
from pySDC.core.Errors import TransferError
from pySDC.core.SpaceTransfer import space_transfer
from pySDC.implementations.datatype_classes.petsc_vec import petsc_vec, petsc_vec_imex, petsc_vec_comp2


class mesh_to_mesh_petsc_dmda(space_transfer):
    """
    This implementation can restrict and prolong between PETSc DMDA grids
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
        super(mesh_to_mesh_petsc_dmda, self).__init__(fine_prob, coarse_prob, params)

        # set interpolation type (no effect as far as I can tell)
        # self.coarse_prob.init.setInterpolationType(PETSc.DMDA.InterpolationType.Q1)
        # define interpolation (only accurate for constant functions)
        self.interp, _ = self.coarse_prob.init.createInterpolation(self.fine_prob.init)
        # define restriction as injection (tranpose of interpolation does not work)
        self.inject = self.coarse_prob.init.createInjection(self.fine_prob.init)

    def restrict(self, F):
        """
        Restriction implementation

        Args:
            F: the fine level data
        """
        if isinstance(F, petsc_vec):
            u_coarse = self.coarse_prob.dtype_u(self.coarse_prob.init)
            self.inject.mult(F, u_coarse)
        elif isinstance(F, petsc_vec_imex):
            u_coarse = self.coarse_prob.dtype_f(self.coarse_prob.init)
            self.inject.mult(F.impl, u_coarse.impl)
            self.inject.mult(F.expl, u_coarse.expl)
        elif isinstance(F, petsc_vec_comp2):
            u_coarse = self.coarse_prob.dtype_f(self.coarse_prob.init)
            self.inject.mult(F.comp1, u_coarse.comp1)
            self.inject.mult(F.comp2, u_coarse.comp2)
        else:
            raise TransferError('Unknown type of fine data, got %s' % type(F))

        return u_coarse

    def prolong(self, G):
        """
        Prolongation implementation

        Args:
            G: the coarse level data
        """
        if isinstance(G, petsc_vec):
            u_fine = self.fine_prob.dtype_u(self.fine_prob.init)
            self.interp.mult(G, u_fine)
        elif isinstance(G, petsc_vec_imex):
            u_fine = self.fine_prob.dtype_f(self.fine_prob.init)
            self.interp.mult(G.impl, u_fine.impl)
            self.interp.mult(G.expl, u_fine.expl)
        elif isinstance(G, petsc_vec_comp2):
            u_fine = self.fine_prob.dtype_f(self.fine_prob.init)
            self.interp.mult(G.comp1, u_fine.comp1)
            self.interp.mult(G.comp2, u_fine.comp2)
        else:
            raise TransferError('Unknown type of coarse data, got %s' % type(G))

        return u_fine
