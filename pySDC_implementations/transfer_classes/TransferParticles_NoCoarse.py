from pySDC_core.SpaceTransfer import space_transfer
from pySDC_implementations.datatype_classes.particles import particles, fields
from pySDC_core.Errors import TransferError


class particles_to_particles(space_transfer):
    """
    Custon transfer class, implements SpaceTransfer.py

    This implementation is just a dummy for particles with no direct functionality, i.e. the number of particles is not
    reduced on the coarse problem
    """

    def __init__(self, fine_prob, coarse_prob, params):
        """
        Initialization routine

        Args:
            fine_prob: fine problem
            coarse_prob: coarse problem
            params: parameters for the transfer operators
        """
        super(particles_to_particles, self).__init__(fine_prob, coarse_prob, params)
        pass

    def restrict(self, F):
        """
        Dummy restriction routine

        Args:
            F: the fine level data
        """

        if isinstance(F, particles):
            G = particles(F)
        elif isinstance(F, fields):
            G = fields(F)
        else:
            raise TransferError("Unknown type of fine data, got %s" % type(F))
        return G

    def prolong(self, G):
        """
        Dummy prolongation routine

        Args:
            G: the coarse level data
        """

        if isinstance(G, particles):
            F = particles(G)
        elif isinstance(G, fields):
            F = fields(G)
        else:
            raise TransferError("Unknown type of coarse data, got %s" % type(G))
        return F
