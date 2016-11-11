from __future__ import division
import numpy as np
import scipy.interpolate as intpl

from pySDC_core.Collocation import CollBase
from pySDC_core.Errors import CollocationError


class EquidistantSpline_Right(CollBase):
    """
    Implements equidistant nodes with right end point included and spline interpolation

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
        super(EquidistantSpline_Right, self).__init__(num_nodes, tleft, tright)
        if num_nodes < 2:
            raise CollocationError("Number of nodes should be at least 2 for equidist. splines, but is %d" % num_nodes)
        # This is a fixed order since we are using splines here! No spectral accuracy!
        self.order = min(num_nodes - 1, 3)  # We need: 1<=order<=5 and order < num_nodes
        self.nodes = self._getNodes
        self.weights = self._getWeights(tleft, tright)
        self.Qmat = self._gen_Qmatrix
        self.Smat = self._gen_Smatrix
        self.delta_m = self._gen_deltas
        self.left_is_node = False
        self.right_is_node = True

    @property
    def _getNodes(self):
        """
        Compute equidistant nodes with right end point included

        Returns:
            np.ndarray: array of equidistant nodes
        """
        return np.linspace(self.tleft + 1.0 / self.num_nodes, self.tright, self.num_nodes, endpoint=True)

    def _getWeights(self, a, b):
        """
        Computes weights using spline interpolation instead of Gaussian quadrature

        Args:
            a (float): left interval boundary
            b (float): right interval boundary

        Returns:
            np.ndarray: weights of the collocation formula given by the nodes
        """

        # get the defining tck's for each spline basis function
        circ_one = np.zeros(self.num_nodes)
        circ_one[0] = 1.0
        tcks = []
        for i in range(self.num_nodes):
            tcks.append(
                intpl.splrep(self.nodes, np.roll(circ_one, i), xb=self.tleft, xe=self.tright, k=self.order, s=0.0))

        weights = np.zeros(self.num_nodes)
        for i in range(self.num_nodes):
            weights[i] = intpl.splint(a, b, tcks[i])

        return weights
