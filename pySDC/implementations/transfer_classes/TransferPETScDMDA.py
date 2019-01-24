
from pySDC.core.Errors import TransferError
from pySDC.core.SpaceTransfer import space_transfer
from pySDC.implementations.datatype_classes.petsc_dmda_grid import petsc_data, rhs_imex_petsc_data, rhs_2comp_petsc_data


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
        if isinstance(F, petsc_data):
            u_coarse = self.coarse_prob.dtype_u(self.coarse_prob.init)
            self.inject.mult(F.values, u_coarse.values)
        elif isinstance(F, rhs_imex_petsc_data):
            u_coarse = self.coarse_prob.dtype_f(self.coarse_prob.init)
            self.inject.mult(F.impl.values, u_coarse.impl.values)
            self.inject.mult(F.expl.values, u_coarse.expl.values)
        elif isinstance(F, rhs_2comp_petsc_data):
            u_coarse = self.coarse_prob.dtype_f(self.coarse_prob.init)
            self.inject.mult(F.comp1.values, u_coarse.comp1.values)
            self.inject.mult(F.comp2.values, u_coarse.comp2.values)
        else:
            raise TransferError('Unknown type of fine data, got %s' % type(F))

        return u_coarse

    def prolong(self, G):
        """
        Prolongation implementation

        Args:
            G: the coarse level data
        """
        if isinstance(G, petsc_data):
            u_fine = self.fine_prob.dtype_u(self.fine_prob.init)
            self.interp.mult(G.values, u_fine.values)
        elif isinstance(G, rhs_imex_petsc_data):
            u_fine = self.fine_prob.dtype_f(self.fine_prob.init)
            self.interp.mult(G.impl.values, u_fine.impl.values)
            self.interp.mult(G.expl.values, u_fine.expl.values)
        elif isinstance(G, rhs_2comp_petsc_data):
            u_fine = self.fine_prob.dtype_f(self.fine_prob.init)
            self.interp.mult(G.comp1.values, u_fine.comp1.values)
            self.interp.mult(G.comp2.values, u_fine.comp2.values)
        else:
            raise TransferError('Unknown type of coarse data, got %s' % type(G))

        return u_fine
