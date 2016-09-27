from __future__ import division
import numpy as np
import scipy.interpolate as intpl

from pySDC.Collocation import CollBase


class EquidistantSpline_Right(CollBase):
    """
    Implements equidistant nodes with left boundary point excluded
    """
    def __init__(self, num_nodes, tleft, tright):
        super(EquidistantSpline_Right, self).__init__(num_nodes, tleft, tright)
        assert num_nodes > 1, "Number of nodes should be at least 1 for EquidistantSpline_Right, but is %d" % num_nodes
        # This is a fixed order since we are using splines here! No spectral accuracy!
        # We need: 1<=order<=5 and order < num_nodes
        self.order = min(num_nodes-1,3)
        self.nodes = self._getNodes
        self.weights = self._getWeights(tleft,tright)
        self.Qmat = self._gen_Qmatrix
        self.Smat = self._gen_Smatrix
        self.delta_m = self._gen_deltas
        self.left_is_node = False
        self.right_is_node = True

    @property
    def _getNodes(self):
        """
        Compute equidistant nodes with right end point included
        :return: list of nodes
        """
        return np.linspace(self.tleft + 1.0 / self.num_nodes, self.tright, self.num_nodes, endpoint=True)

    def _getWeights(self, a, b):
        """
        Computes weights using spline interpolation instead of Gaussian quadrature
        :param a: left interval boundary
        :param b: right interval boundary
        :return: weights of the collocation formula given by the nodes
        """

        # get the defining tck's for each spline basis function
        circ_one = np.zeros(self.num_nodes)
        circ_one[0] = 1.0
        tcks = []
        for i in range(self.num_nodes):
            tcks.append(intpl.splrep(self.nodes, np.roll(circ_one, i), xb=self.tleft, xe=self.tright, k=self.order, s=0.0))

        weights = np.zeros(self.num_nodes)
        for i in range(self.num_nodes):
            weights[i] = intpl.splint(a, b, tcks[i])

        return weights

