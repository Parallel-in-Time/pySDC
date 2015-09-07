from __future__ import division
import numpy as np
import scipy.sparse as sprs
from pySDC.Transfer import transfer
from pySDC.datatype_classes.mesh import mesh, rhs_imex_mesh
from pySDC.tools.transfer_tools import *


class value_to_value(transfer):
    """
    Transferclass, which does nothing more as map the value onto the value, because
    its a scalar problem.
    """

    def __init__(self, fine_level, coarse_level,*args,**kwargs):
        print args
        if len(args) < 4:
            super(value_to_value,self).__init__(fine_level, coarse_level, *args)
        else:
            args_alt = {}

        super(value_to_value,self).__init__(fine_level, coarse_level, *args)
        print vars(self)
        self.Rspace = np.eye(1)
        self.Pspace = np.eye(1)

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
