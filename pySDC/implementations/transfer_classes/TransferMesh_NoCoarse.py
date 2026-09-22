from pySDC.core.space_transfer import SpaceTransfer


class mesh_to_mesh(SpaceTransfer):
    """
    Custom base_transfer class, implements Transfer.py

    This implementation is for identical fine and coarse meshes: both directions just copy the data.
    """

    def restrict(self, F):
        """
        Restriction implementation

        Args:
            F: the fine level data (easier to access than via the fine attribute)
        """
        return type(F)(F)

    def prolong(self, G):
        """
        Prolongation implementation

        Args:
            G: the coarse level data (easier to access than via the coarse attribute)
        """
        return type(G)(G)
