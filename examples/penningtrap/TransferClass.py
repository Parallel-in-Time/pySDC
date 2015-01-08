from pySDC.Transfer import transfer
from pySDC.datatype_classes.particles import particles

class particles_to_particles(transfer):
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

    def __init__(self,fine_level,coarse_level):
        """
        Initialization routine

        Args:
            fine_level: fine level connected with the transfer operations (passed to parent)
            coarse_level: coarse level connected with the transfer operations (passed to parent)
        """
        super(particles_to_particles,self).__init__(fine_level,coarse_level)
        pass

    def restrict_space(self,F):
        """
        Dummy restriction routine

        Args:
            F: the fine level data (easier to access than via the fine attribute)

        """

        G = particles(F)
        return G

    def prolong_space(self,G):
        """
        Dummy prolongation routine

        Args:
            G: the coarse level data (easier to access than via the coarse attribute)
        """

        F = particles(G)
        return F