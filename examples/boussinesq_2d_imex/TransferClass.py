from pySDC.Transfer import transfer
from pySDC.datatype_classes.mesh import mesh, rhs_imex_mesh

class mesh_to_mesh_2d(transfer):
    """
    Custon transfer class, implements Transfer.py

    This implementation is just a dummy for particles with no functionality. It can be used to check if in the particle
    setups the number of iterations is halved once two levels are used.

    Attributes:
        fine: reference to the fine level
        coarse: reference to the coarse level
        init_f: number of variables on the fine level (whatever init represents there)
        init_c: number of variables on the coarse level (whatever init represents there)
    """

    def __init__(self,fine_level,coarse_level,params):
        """
        Initialization routine

        Args:
            fine_level: fine level connected with the transfer operations (passed to parent)
            coarse_level: coarse level connected with the transfer operations (passed to parent)
        """
        super(mesh_to_mesh_2d,self).__init__(fine_level,coarse_level,params)
        pass

    def restrict_space(self,F):
        """
        Dummy restriction routine

        Args:
            F: the fine level data (easier to access than via the fine attribute)

        """

        if isinstance(F,mesh):
            G = mesh(F)
        elif isinstance(F,rhs_imex_mesh):
            G = rhs_imex_mesh(F)
        else:
            print('Transfer error')
            exit()
        return G

    def prolong_space(self,G):
        """
        Dummy prolongation routine

        Args:
            G: the coarse level data (easier to access than via the coarse attribute)
        """

        if isinstance(G,mesh):
            F = mesh(G)
        elif isinstance(G,rhs_imex_mesh):
            F = rhs_imex_mesh(G)
        else:
            print('Transfer error')
            exit()
        return F