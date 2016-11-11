from __future__ import division
import numpy as np
import numpy.polynomial.legendre as leg

from pySDC_core.Collocation import CollBase
from pySDC_core.Errors import CollocationError

class CollGaussLobatto(CollBase):
    """
    Implements Gauss-Lobatto Quadrature

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

    def __init__(self, num_nodes, tleft, tright):
        """
        Initialization

        Args:
            num_nodes (int): number of nodes
            tleft (float): left interval boundary (usually 0)
            tright (float): right interval boundary (usually 1)
        """
        super(CollGaussLobatto, self).__init__(num_nodes, tleft, tright)
        if num_nodes < 2:
            raise CollocationError("Number of nodes should be at least 2 for Gauss-Lobatto, but is %d" % num_nodes)
        self.order = 2 * self.num_nodes - 2
        self.nodes = self._getNodes
        self.weights = self._getWeights(tleft, tright)
        self.Qmat = self._gen_Qmatrix
        self.Smat = self._gen_Smatrix
        self.delta_m = self._gen_deltas
        self.left_is_node = True
        self.right_is_node = True

    @property
    def _getNodes(self):
        """
        Computes Gauss-Lobatto integration nodes.

        Calculates the Gauss-Lobatto integration nodes via a root calculation of derivatives of the legendre
        polynomials. Note that the precision of float 64 is not guarantied.

        Copyright by Dieter Moser, 2014

        Returns:
            np.ndarray: array of Gauss-Lobatto nodes
        """

        M = self.num_nodes
        a = self.tleft
        b = self.tright

        roots = leg.legroots(leg.legder(np.array([0] * (M - 1) + [1], dtype=np.float64)))
        nodes = np.array(np.append([-1.0], np.append(roots, [1.0])), dtype=np.float64)

        nodes = (a * (1 - nodes) + b * (1 + nodes)) / 2

        return nodes
