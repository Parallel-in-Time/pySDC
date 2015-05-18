from __future__ import division
import numpy as np

from pySDC.Transfer import transfer
from pySDC.datatype_classes.mesh import mesh, rhs_imex_mesh

# FIXME: extend this to ndarrays
class mesh_to_mesh_1d_periodic(transfer):
    """
    Custon transfer class, implements Transfer.py

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

    def __init__(self,fine_level,coarse_level,*args):
        """
        Initialization routine

        Args:
            fine_level: fine level connected with the transfer operations (passed to parent)
            coarse_level: coarse level connected with the transfer operations (passed to parent)
        """

        # invoke super initialization
        super(mesh_to_mesh_1d_periodic,self).__init__(fine_level,coarse_level)

        # if number of variables is the same on both levels, Rspace and Pspace are identity
        if self.init_c == self.init_f:
            self.Rspace = np.eye(self.init_c)
        # assemble weighted restriction by hand
        else:
            self.Rspace = np.zeros((self.init_c,self.init_f))
            np.fill_diagonal(self.Rspace[:,0::2],1)
            np.fill_diagonal(self.Rspace[:,1::2],1/2)
            np.fill_diagonal(self.Rspace[1:,1::2],1/2)
            self.Rspace[0,-1] = 1/2
            self.Rspace = 1/2*self.Rspace

        # if number of variables is the same on both levels, Rspace and Pspace are identity
        if self.init_f == self.init_c:
            self.Pspace = np.eye(self.init_f)
        # assemble 7th-order prolongation by hand
        else:
            self.Pspace = np.zeros((self.init_f,self.init_c))

            np.fill_diagonal(self.Pspace[0::2,:],1)
            np.fill_diagonal(self.Pspace[1::2,:],0.5859375)
            np.fill_diagonal(self.Pspace[1::2,1:],0.5859375)
            np.fill_diagonal(self.Pspace[3::2,:],-0.09765625)
            np.fill_diagonal(self.Pspace[1::2,2:],-0.09765625)
            np.fill_diagonal(self.Pspace[5::2,:],0.01171875)
            np.fill_diagonal(self.Pspace[1::2,3:],0.01171875)
            self.Pspace[1,-1] = -0.09765625
            self.Pspace[1,-2] = 0.01171875
            self.Pspace[3,-1] = 0.01171875
            self.Pspace[-1,0] = 0.5859375
            self.Pspace[-1,1] = -0.09765625
            self.Pspace[-1,2] = 0.01171875
            self.Pspace[-3,0] = -0.09765625
            self.Pspace[-3,1] = 0.01171875
            self.Pspace[-5,0] = 0.01171875

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
