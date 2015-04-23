# coding=utf-8
from __future__ import division
import numpy as np
import scipy.interpolate as intpl
from pySDC.Transfer import transfer
from pySDC.datatype_classes.mesh import mesh, rhs_imex_mesh
from pySDC.tools.transfer_tools import *

class mesh_to_mesh(transfer):
    """
    Custom transfer class, implements Transfer.py

    This implementation can restrict and prolong between meshes of arbitrary dimension,
    through the use of kronecker products and interpolation and restriction matrices
    for each direction.

    Attributes:
        fine: reference to the fine level
        coarse: reference to the coarse level
        fine_mesh: mesh of the fine level
        coarse_mesh: mesh of the coarse level
        Rspace: spatial restriction matrix, dim. Nf_1*Nf_2*...Nf_D x Nc_1*Nc_2*...*Nc_D
        Pspace: spatial prolongation matrix, dim. Nc_1*Nc_2*...Nc_D x Nf_1*Nf_2*...*Nf_D
    """

    def __init__(self, fine_level, coarse_level, interpolation_order=[2], restriction_order=[2], sparse_format="dense"):
        """
        Initialization routine

        Args:
            fine_level: fine level connected with the transfer operations (passed to parent)
            coarse_level: coarse level connected with the transfer operations (passed to parent)
            interpolation_order:list of number of points used for the interpolation matrix for each direction
            restriction_order:list of number of points used for the restriction matrix for each direction
            return_type:which kind of sparsity is used
        """


        # invoke super initialization
        super(mesh_to_mesh, self).__init__(fine_level, coarse_level)

        # get the meshes, get_mesh() is only part of special problem classes
        self.coarse_mesh = [coarse_level.sweep.coll.nodes, coarse_level.prob.get_mesh("list")]
        self.fine_mesh = [fine_level.sweep.coll.nodes, fine_level.prob.get_mesh("list")]

        interpolation_matrix_list = []
        restriction_matrix_list = []
        len_intpl_order = len(interpolation_order)
        len_restr_order = len(restriction_order)

        # build the matrices for each direction of the mesh
        for c_mesh, f_mesh, i in zip(self.coarse_mesh, self.fine_mesh, range(len(self.fine_mesh))):
            interpolation_matrix_list.append(
                interpolation_matrix_1d(f_mesh, c_mesh, interpolation_order[i % len_intpl_order], sparse_format))
            restriction_matrix_list.append(
                restriction_matrix_1d(f_mesh, c_mesh, restriction_order[i % len_restr_order], sparse_format))

        # using the kronecker product to assemble the whole interpolation and restriction matrix
        self.Pspace = kron_on_list(interpolation_matrix_list)
        self.Rspace = kron_on_list(restriction_matrix_list)

    def restrict_space(self,F):
        """
        Restriction implementation

        Args:
            F: the fine level data (easier to access than via the fine attribute)
        """

        if isinstance(F,mesh):
            u_coarse = mesh(self.init_c,val=0)
            u_coarse.values = np.dot(self.Rspace,F.values)
        elif isinstance(F,rhs_imex_mesh):
            u_coarse = rhs_imex_mesh(self.init_c)
            u_coarse.impl.values = np.dot(self.Rspace,F.impl.values)
            u_coarse.expl.values = np.dot(self.Rspace,F.expl.values)

        return u_coarse

    def prolong_space(self,G):
        """
        Prolongation implementation

        Args:
            G: the coarse level data (easier to access than via the coarse attribute)
        """

        if isinstance(G,mesh):
            u_fine = mesh(self.init_c,val=0)
            u_fine.values = np.dot(self.Pspace,G.values)
        elif isinstance(G,rhs_imex_mesh):
            u_fine = rhs_imex_mesh(self.init_c)
            u_fine.impl.values = np.dot(self.Pspace,G.impl.values)
            u_fine.expl.values = np.dot(self.Pspace,G.expl.values)

        return u_fine


class composed_mesh_to_composed_mesh(transfer):
    """
    Takes a list of mesh_to_mesh_1d classes and composes them into one class
    """

    def __init__(self, fine_level, coarse_level, mesh_to_mesh_1d_list, sparse_format="dense"):

        # invoke super initialization
        super(composed_mesh_to_composed_mesh, self).__init__(fine_level, coarse_level)
        self.sparse_format = sparse_format
        self.Pspace = to_sparse(reduce(lambda x, y: sprs.kron(x.Pspace, y.Pspace), mesh_to_mesh_1d_list),
                                self.sparse_format)
        self.Rspace = to_sparse(reduce(lambda x, y: sprs.kron(x.Rspace, y.Rspace), mesh_to_mesh_1d_list),
                                self.sparse_format)

    def restrict_space(self,F):
        """
        Restriction implementation

        Args:
            F: the fine level data (easier to access than via the fine attribute)
        """

        if isinstance(F, mesh):
            u_coarse = mesh(self.init_c, val=0)
            u_coarse.values = np.dot(self.Rspace, F.values)
        elif isinstance(F, rhs_imex_mesh):
            u_coarse = rhs_imex_mesh(self.init_c)
            u_coarse.impl.values = np.dot(self.Rspace, F.impl.values)
            u_coarse.expl.values = np.dot(self.Rspace, F.expl.values)

        return u_coarse

    def prolong_space(self,G):
        """
        Prolongation implementation

        Args:
            G: the coarse level data (easier to access than via the coarse attribute)
        """

        if isinstance(G,mesh):
            u_fine = mesh(self.init_c,val=0)
            u_fine.values = np.dot(self.Pspace,G.values)
        elif isinstance(G,rhs_imex_mesh):
            u_fine = rhs_imex_mesh(self.init_c)
            u_fine.impl.values = np.dot(self.Pspace,G.impl.values)
            u_fine.expl.values = np.dot(self.Pspace,G.expl.values)

        return u_fine

class time_mesh_to_time_mesh(transfer):
    """ Takes two levels and constructs the interpolation and restriction matrices between
        just the temporal part.
    """
    def __init__(self, fine_level, coarse_level, interpolation_order=2, restriction_order=2, sparse_format="dense"):
        """

        :param fine_level:
        :param coarse_level:
        :param interpolation_order:
        :param restriction_order:
        :param sparse_format:
        :return:
        """

        super(time_mesh_to_time_mesh, self).__init__(fine_level, coarse_level)

        # get the meshes, get_mesh() is only part of special problem classes and get_nodes is part of a special
        # sweeper class
        self.coarse_mesh = coarse_level.sweep.coll.nodes
        self.fine_mesh = fine_level.sweep.coll.nodes
        self.sparse_format = sparse_format
        self.Pspace = interpolation_matrix_1d(self.fine_mesh, self.coarse_mesh, interpolation_order, sparse_format)
        self.Rspace = restriction_matrix_1d(self.fine_mesh, self.coarse_mesh, restriction_order, sparse_format)


    def restrict_space(self, F):
        """
        Restriction implementation

        Args:
            F: the fine level data (easier to access than via the fine attribute)
        """

        if isinstance(F,mesh):
            u_coarse = mesh(self.init_c,val=0)
            u_coarse.values = np.dot(self.Rspace, F.values)
        elif isinstance(F,rhs_imex_mesh):
            u_coarse = rhs_imex_mesh(self.init_c)
            u_coarse.impl.values = np.dot(self.Rspace, F.impl.values)
            u_coarse.expl.values = np.dot(self.Rspace, F.expl.values)

        return u_coarse

    def prolong_space(self, G):
        """
        Prolongation implementation

        Args:
            G: the coarse level data (easier to access than via the coarse attribute)
        """

        if isinstance(G, mesh):
            u_fine = mesh(self.init_c, val=0)
            u_fine.values = np.dot(self.Pspace, G.values)
        elif isinstance(G, rhs_imex_mesh):
            u_fine = rhs_imex_mesh(self.init_c)
            u_fine.impl.values = np.dot(self.Pspace, G.impl.values)
            u_fine.expl.values = np.dot(self.Pspace, G.expl.values)

        return u_fine
