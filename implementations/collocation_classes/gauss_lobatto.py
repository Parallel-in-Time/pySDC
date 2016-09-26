from __future__ import division
import numpy as np
import numpy.polynomial.legendre as leg

from pySDC.Collocation import CollBase


class CollGaussLobatto(CollBase):
    """
    Implements Gauss-Lobatto Quadrature by deriving from CollBase and implementing Gauss-Lobatto nodes
    """
    def __init__(self, num_nodes, tleft, tright):
        super(CollGaussLobatto, self).__init__(num_nodes, tleft, tright)
        assert num_nodes >= 2, "Number of nodes should be at least 2 for Gauss-Lobatto, but is %d" % num_nodes
        self.order = 2 * self.num_nodes - 2
        self.nodes = self._getNodes
        self.weights = self._getWeights(tleft,tright)
        self.Qmat = self._gen_Qmatrix
        self.Smat = self._gen_Smatrix
        self.delta_m = self._gen_deltas
        self.left_is_node = True
        self.right_is_node = True

    @property
    def _getNodes(self):
        """
        Copyright by Dieter Moser, 2014
        Computes Gauss-Lobatto integration nodes.

        Calculates the Gauss-Lobatto integration nodes via a root calculation of derivatives of the legendre
        polynomials. Note that the precision of float 64 is not guarantied.
        """
        M = self.num_nodes
        a = self.tleft
        b = self.tright

        roots = leg.legroots(leg.legder(np.array([0] * (M - 1) + [1], dtype=np.float64)))
        nodes = np.array(np.append([-1.0], np.append(roots, [1.0])), dtype=np.float64)

        nodes = (a * (1 - nodes) + b * (1 + nodes)) / 2

        return nodes