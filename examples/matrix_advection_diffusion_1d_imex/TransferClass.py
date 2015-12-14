from __future__ import division
import numpy as np
from pySDC.Transfer import transfer
from pySDC.datatype_classes.mesh import mesh, rhs_imex_mesh
from pySDC.tools.transfer_tools import *

# FIXME: extend this to ndarrays
class mesh_to_mesh_1d(transfer):
    """
    Custom transfer class, implements Transfer.py

    This implementation can restrict and prolong between 1d meshes, using weigthed restriction and 7th-order prologation
    via matrix-vector multiplication.

    Attributes:
        fine: reference to the fine level
        coarse: reference to the coarse level
        init_f: number of variables on the fine level (whatever init represents there)
        init_c: number of variables on the coarse level (whatever init represents there)
        Rspace: spatial restriction matrix, dim. Nf x Nc
        Pspace: spatial prolongation matrix, dim. Nc x Nf
    """

    def __init__(self, fine_level, coarse_level,*args):
        """
        Initialization routine

        Args:
            fine_level: fine level connected with the transfer operations (passed to parent)
            coarse_level: coarse level connected with the transfer operations (passed to parent)
        """

        # invoke super initialization
        super(mesh_to_mesh_1d,self).__init__(fine_level, coarse_level)

        # if number of variables is the same on both levels, Rspace and Pspace are identity
        if self.init_c == self.init_f:
            self.Rspace = np.eye(self.init_c)
        # assemble weighted restriction by hand
        else:
            self.Rspace = np.zeros((self.init_f,self.init_c))
            np.fill_diagonal(self.Rspace[1::2,:],1)
            np.fill_diagonal(self.Rspace[0::2,:],1/2)
            np.fill_diagonal(self.Rspace[2::2,:],1/2)
            self.Rspace = 1/2*self.Rspace.T

        # if number of variables is the same on both levels, Rspace and Pspace are identity
        if self.init_f == self.init_c:
            self.Pspace = np.eye(self.init_f)
        # assemble 7th-order prolongation by hand
        else:
            self.Pspace = np.zeros((self.init_f,self.init_c))
            np.fill_diagonal(self.Pspace[1::2,:],1)
            np.fill_diagonal(self.Pspace[0::2,:],1/2)
            np.fill_diagonal(self.Pspace[2::2,:],1/2)

            # this would be 3rd-order accurate
            # c1 = -0.0625
            # c2 = 0.5625
            # c3 = c2
            # c4 = c1
            # np.fill_diagonal(self.Pspace[0::2,:],c3)
            # np.fill_diagonal(self.Pspace[2::2,:],c2)
            # np.fill_diagonal(self.Pspace[0::2,1:],c4)
            # np.fill_diagonal(self.Pspace[4::2,:],c1)
            # self.Pspace[0,0:3] = [0.9375, -0.3125, 0.0625]
            # self.Pspace[-1,-3:init_c] = [0.0625, -0.3125, 0.9375]

            np.fill_diagonal(self.Pspace[0::2,:],0.5859375)
            np.fill_diagonal(self.Pspace[2::2,:],0.5859375)
            np.fill_diagonal(self.Pspace[0::2,1:],-0.09765625)
            np.fill_diagonal(self.Pspace[4::2,:],-0.09765625)
            np.fill_diagonal(self.Pspace[0::2,2:],0.01171875)
            np.fill_diagonal(self.Pspace[6::2,:],0.01171875)
            self.Pspace[0,0:5] = [1.23046875, -0.8203125, 0.4921875, -0.17578125, 0.02734375]
            self.Pspace[2,0:5] = [0.41015625, 0.8203125, -0.2734375, 0.08203125, -0.01171875]
            self.Pspace[-1,-5:self.init_c] = [0.02734375,  -0.17578125, 0.4921875, -0.8203125, 1.23046875]
            self.Pspace[-3,-5:self.init_c] = [-0.01171875, 0.08203125, -0.2734375, 0.8203125, 0.41015625]

        pass

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

# So wird diese classe aufgerufen, d.h. die args
#
#    space_transfer_list = map(lambda x, int_ord, restr_ord: transfer_class(x.levels[0], x.levels[-1],
#                                                                           int_ord, restr_ord, kwargs['sparse_format']),



class mesh_to_mesh_1d_periodic(transfer):
    """
    Custom transfer class, implements Transfer.py

    This implementation can restrict and prolong between 1d meshes, using arbitrary order interpolation and the restriction given as P^T
    via matrix-vector multiplication.

    Attributes:
        fine: reference to the fine level
        coarse: reference to the coarse level
        init_f: number of variables on the fine level (whatever init represents there)
        init_c: number of variables on the coarse level (whatever init represents there)
        Rspace: spatial restriction matrix, dim. Nf x Nc
        Pspace: spatial prolongation matrix, dim. Nc x Nf
    """

    def __init__(self, fine_level, coarse_level,*args,**kwargs):
        """
        Initialization routine

        Args:
            fine_level: fine level connected with the transfer operations (passed to parent)
            coarse_level: coarse level connected with the transfer operations (passed to parent)
        """
        # invoke super initialization
        super(mesh_to_mesh_1d_periodic,self).__init__(fine_level, coarse_level)
        if len(args) == 0:
            self.int_ord = 8
            self.restr_ord = 2
            self.sparse_format = "csc"
        else:
            self.int_ord = args[0]
            self.restr_ord = args[1]
            self.sparse_format = args[2]

        # if number of variables is the same on both levels, Rspace and Pspace are identity
        if self.init_c == self.init_f:
            self.Rspace = sprs.diags(np.ones(self.init_c))
            self.Pspace = sprs.diags(np.ones(self.init_c))
        # assemble weighted restriction by hand
        else:
        # def restriction_matrix_1d(fine_grid, coarse_grid, k=2, return_type="csc", periodic=False, T=1.0):
        # def interpolation_matrix_1d(fine_grid, coarse_grid, k=2, return_type="csc", periodic=False, T=1.0):
            f_grid = fine_level.prob.get_mesh("meshgrid")
            c_grid = coarse_level.prob.get_mesh("meshgrid")
            #   print f_grid
            #   print c_grid
            #   print self.restr_ord
            #   print self.int_ord
            self.Rspace = interpolation_matrix_1d(f_grid, c_grid, self.restr_ord, self.sparse_format, True, 1.0).transpose()/2.0
            self.Pspace = interpolation_matrix_1d(f_grid, c_grid, self.int_ord, self.sparse_format, True, 1.0)

    def restrict_space(self,F):
        """
        Restriction implementation

        Args:
            F: the fine level data (easier to access than via the fine attribute)
        """

        if isinstance(F,mesh):
            u_coarse = mesh(self.init_c,val=0)
            u_coarse.values = self.Rspace.dot(F.values)
        elif isinstance(F,rhs_imex_mesh):
            u_coarse = rhs_imex_mesh(self.init_c)
            u_coarse.impl.values = self.Rspace.dot(F.impl.values)
            u_coarse.expl.values = self.Rspace.dot(F.expl.values)

        return u_coarse

    def prolong_space(self,G):
        """
        Prolongation implementation

        Args:
            G: the coarse level data (easier to access than via the coarse attribute)
        """

        if isinstance(G,mesh):
            u_fine = mesh(self.init_c,val=0)
            u_fine.values = self.Pspace.dot(G.values)
        elif isinstance(G,rhs_imex_mesh):
            u_fine = rhs_imex_mesh(self.init_c)
            u_fine.impl.values = self.Pspace.dot(G.impl.values)
            u_fine.expl.values = self.Pspace.dot(G.expl.values)

        return u_fine

