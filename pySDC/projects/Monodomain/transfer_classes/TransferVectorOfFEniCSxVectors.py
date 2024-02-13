import mpi4py

mpi4py.rc.recv_mprobe = False
import dolfinx as df
from petsc4py import PETSc
import numpy as np
from pySDC.core.SpaceTransfer import space_transfer
from dolfinx.cpp.fem.petsc import interpolation_matrix


class TransferVectorOfFEniCSxVectors(space_transfer):
    """
    This implementation can restrict and prolong between super vectors
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
        super(TransferVectorOfFEniCSxVectors, self).__init__(fine_prob, coarse_prob, params)

        # padding = 1e-14

        self.restrict_interpolation_data = df.fem.create_nonmatching_meshes_interpolation_data(
            coarse_prob.parabolic.domain._cpp_object, coarse_prob.parabolic.V.element, fine_prob.parabolic.domain._cpp_object
        )
        self.prolong_interpolation_data = df.fem.create_nonmatching_meshes_interpolation_data(
            fine_prob.parabolic.domain._cpp_object, fine_prob.parabolic.V.element, coarse_prob.parabolic.domain._cpp_object
        )

        self.coarse_mesh = coarse_prob.parabolic.domain
        self.coarse_map = self.coarse_mesh.topology.index_map(self.coarse_mesh.topology.dim)
        self.coarse_cells = np.arange(self.coarse_map.size_local + self.coarse_map.num_ghosts, dtype=np.int32)

        self.fine_mesh = fine_prob.parabolic.domain
        self.fine_map = self.fine_mesh.topology.index_map(self.fine_mesh.topology.dim)
        self.fine_cells = np.arange(self.fine_map.size_local + self.fine_map.num_ghosts, dtype=np.int32)

        # self.matrix_coarse_to_fine = interpolation_matrix(coarse_prob.parabolic.V._cpp_object, fine_prob.parabolic.V._cpp_object)
        # self.matrix_coarse_to_fine.assemble()

        # self.matrix_fine_to_coarse = interpolation_matrix(fine_prob.parabolic.V._cpp_object, coarse_prob.parabolic.V._cpp_object)
        # self.matrix_fine_to_coarse.assemble()

    def restrict(self, F):
        """
        Restriction implementation

        Args:
            F: the fine level data
        """
        u_coarse = type(F)(init=self.coarse_prob.init, val=1.0, type_sub_vector=self.coarse_prob.vector_type, size=self.coarse_prob.size)
        F.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

        u_coarse.restrict(F, cells=self.coarse_cells, nmm_interpolation_data=self.restrict_interpolation_data)

        return u_coarse

    def prolong(self, G):
        """
        Prolongation implementation

        Args:
            G: the coarse level data
        """
        u_fine = type(G)(init=self.fine_prob.init, val=1.0, type_sub_vector=self.fine_prob.vector_type, size=self.fine_prob.size)
        G.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

        u_fine.prolong(G, cells=self.fine_cells, nmm_interpolation_data=self.prolong_interpolation_data)

        return u_fine
