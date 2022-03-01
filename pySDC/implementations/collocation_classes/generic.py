from pySDC.core.Collocation import CollBase
from pySDC.core.Nodes import NodesGenerator

import numpy as np
import scipy.interpolate as intpl

NODE_TYPES = ['EQUID', 'LEGENDRE',
              'CHEBY-1', 'CHEBY-2', 'CHEBY-3', 'CHEBY-4']

QUAD_TYPES = ['GAUSS', 'RADAU-LEFT', 'RADAU-RIGHT', 'LOBATTO']

class Collocation(CollBase):

    def __init__(self, num_nodes, tleft, tright,
                 node_type='LEGENDRE', quad_type='LOBATTO', useSpline=False):
        # Base constructor
        super(Collocation, self).__init__(num_nodes, tleft, tright)
        # Instanciate attributes
        self.nodeGenerator = NodesGenerator(node_type, quad_type)
        if useSpline:
            self._getWeights = self._getWeights_spline
            # We need: 1<=order<=5 and order < num_nodes
            self.order = min(num_nodes - 1, 3)
        elif quad_type == 'GAUSS':
            self.order = 2 * num_nodes
        elif quad_type.startswith('RADAU'):
            self.order = 2 * num_nodes - 1
        elif quad_type == 'LOBATTO':
            self.order = 2 * num_nodes - 2
        self.nodes = self._getNodes
        self.weights = self._getWeights(tleft, tright)
        self.Qmat = self._gen_Qmatrix
        self.Smat = self._gen_Smatrix
        self.delta_m = self._gen_deltas
        self.left_is_node = False
        self.right_is_node = False

    @property
    def _getNodes(self):
        """
        Computes nodes using an internal NodesGenerator object

        Returns:
            np.ndarray: array of Gauss-Legendre nodes
        """
        # Generate normalized nodes in [-1, 1]
        nodes = self.nodeGenerator.getNodes(self.num_nodes)

        # Scale nodes to [tleft, tright]
        a = self.tleft
        b = self.tright
        nodes += 1
        nodes /= 2
        nodes *= b-a
        nodes += a

        return nodes

    def _getWeights_spline(self, a, b):
        """
        Computes weights using spline interpolation instead of Gaussian
        quadrature

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
