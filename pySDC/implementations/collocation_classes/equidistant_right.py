from __future__ import division
import numpy as np

from pySDC.core.Collocation import CollBase
from pySDC.core.Collocation import CollocationError


class EquidistantNoLeft(CollBase):
    """
    Implements equidistant nodes with left boundary point excluded

    Attributes:
        order (int): order of the quadrature
        num_nodes (int): number of collocation nodes
        tleft (float): left interval point
        tright (float): right interval point
        nodes (numpy.ndarray): array of quadrature nodes
        weights (numpy.ndarray): array of quadrature weights for the full interval
        Qmat (numpy.ndarray): matrix containing the weights for tleft to node
        Smat (numpy.ndarray): matrix containing the weights for node to node
        delta_m (numpy.ndarray): array of distances between nodes
        right_is_node (bool): flag to indicate whether right point is collocation node
        left_is_node (bool): flag to indicate whether left point is collocation node
    """

    def __init__(self, num_nodes, tleft, tright, do_DG):
        """
        Initialization

        Args:
            num_nodes (int): number of nodes
            tleft (float): left interval boundary (usually 0)
            tright (float): right interval boundary (usually 1)
            do_DG (bool): option to use DG-based Q matrix
        """
        super(EquidistantNoLeft, self).__init__(num_nodes, tleft, tright, do_DG)
        if num_nodes < 2:
            raise CollocationError("Number of nodes should be at least 2 for equidistant, but is %d" % num_nodes)
        self.order = self.num_nodes
        self.left_is_node = False
        self.right_is_node = True
        self.nodes = self._getNodes
        self.weights = self._getWeights(tleft, tright)
        self.Qmat = self._gen_Qmatrix
        self.Smat = self._gen_Smatrix
        self.delta_m = self._gen_deltas

    @property
    def _getNodes(self):
        """
        Computes integration nodes with both boundaries excluded

        Returns:
            np.ndarray: array of equidistant nodes
        """
        M = self.num_nodes
        a = self.tleft
        b = self.tright
        nodes = np.linspace(a, b, M + 1)
        nodes = nodes[1:]
        return nodes
