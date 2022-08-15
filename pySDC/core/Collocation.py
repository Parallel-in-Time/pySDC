import logging

import numpy as np
import scipy.interpolate as intpl

from pySDC.core.Nodes import NodesGenerator
from pySDC.core.Errors import CollocationError
from pySDC.core.Lagrange import LagrangeApproximation


class CollBase(object):
    """
    Generic collocation class, that contains everything to do integration over
    intervals and between nodes.
    It can be used to produce many kind of quadrature nodes from various
    distribution (awesome!).

    It is based on the two main parameters that define the nodes :

    - node_type : the node distribution used for the collocation method
    - quad_type : the type of quadrature used (inclusion of not of boundary)

    Current implementation provides the following available parameter values
    for node_type :

    - EQUID : equidistant node distribution
    - LEGENDRE : distribution from Legendre polynomials
    - CHEBY-{1,2,3,4} : distribution from Chebyshev polynomials of a given kind

    The type of quadrature cann be GAUSS (only inner nodes), RADAU-LEFT
    (inclusion of the left boundary), RADAU-RIGHT (inclusion of the right
    boundary) and LOBATTO (inclusion of left and right boundary).

    Furthermore, the ``useSpline`` option can be activated to eventually use
    spline interpolation when computing the weights.

    Here is the equivalency table with the original classes implemented in
    pySDC :

    +-------------------------+-----------+-------------+-----------+
    | Original Class          | node_type | quad_type   | useSpline |
    +=========================+===========+=============+===========+
    | Equidistant             | EQUID     | LOBATTO     | False     |
    +-------------------------+-----------+-------------+-----------+
    | EquidistantInner        | EQUID     | GAUSS       | False     |
    +-------------------------+-----------+-------------+-----------+
    | EquidistantNoLeft       | EQUID     | RADAU-RIGHT | False     |
    +-------------------------+-----------+-------------+-----------+
    | EquidistantSpline_Right | EQUID     | RADAU-RIGHT | True      |
    +-------------------------+-----------+-------------+-----------+
    | CollGaussLegendre       | LEGENDRE  | GAUSS       | False     |
    +-------------------------+-----------+-------------+-----------+
    | CollGaussLobatto        | LEGENDRE  | LOBATTO     | False     |
    +-------------------------+-----------+-------------+-----------+
    | CollGaussRadau_Left     | LEGENDRE  | RADAU-LEFT  | False     |
    +-------------------------+-----------+-------------+-----------+
    | CollGaussRadau_Right    | LEGENDRE  | RADAU-RIGHT | False     |
    +-------------------------+-----------+-------------+-----------+

    Attributes:
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

    def __init__(
        self, num_nodes=None, tleft=0, tright=1, node_type='LEGENDRE', quad_type='LOBATTO', useSpline=False, **kwargs
    ):
        """
        Initialization routine for a collocation object

        Args:
            num_nodes (int): number of collocation nodes
            tleft (float): left interval point
            tright (float): right interval point
        """

        if not num_nodes > 0:
            raise CollocationError('At least one quadrature node required, got %s' % num_nodes)
        if not tleft < tright:
            raise CollocationError('Interval boundaries are corrupt, got %s and %s' % (tleft, tright))

        self.logger = logging.getLogger('collocation')

        # Set number of nodes, left and right interval boundaries
        self.num_nodes = num_nodes
        self.tleft = tleft
        self.tright = tright

        self.node_type = node_type
        self.quad_type = quad_type

        # Instanciate attributes
        self.nodeGenerator = NodesGenerator(self.node_type, self.quad_type)
        if useSpline:
            self._getWeights = self._getWeights_spline
            # We need: 1<=order<=5 and order < num_nodes
            self.order = min(num_nodes - 1, 3)
        elif self.node_type == 'EQUID':
            self.order = num_nodes
        else:
            if self.quad_type == 'GAUSS':
                self.order = 2 * num_nodes
            elif self.quad_type.startswith('RADAU'):
                self.order = 2 * num_nodes - 1
            elif self.quad_type == 'LOBATTO':
                self.order = 2 * num_nodes - 2

        self.left_is_node = self.quad_type in ['LOBATTO', 'RADAU-LEFT']
        self.right_is_node = self.quad_type in ['LOBATTO', 'RADAU-RIGHT']

        self.nodes = self._getNodes
        self.weights = self._getWeights(tleft, tright)
        self.Qmat = self._gen_Qmatrix_spline if useSpline else self._gen_Qmatrix
        self.Smat = self._gen_Smatrix
        self.delta_m = self._gen_deltas

    @staticmethod
    def evaluate(weights, data):
        """
        Evaluates the quadrature over the full interval

        Args:
            weights (numpy.ndarray): array of quadrature weights for the full interval
            data (numpy.ndarray): f(x) to be integrated

        Returns:
            numpy.ndarray: integral over f(x) between tleft and tright
        """
        if not np.size(weights) == np.size(data):
            raise CollocationError("Input size does not match number of weights, but is %s" % np.size(data))

        return np.dot(weights, data)

    def _getWeights(self, a, b):
        """
        Computes weights using barycentric interpolation

        Args:
            a (float): left interval boundary
            b (float): right interval boundary

        Returns:
            numpy.ndarray: weights of the collocation formula given by the nodes
        """
        if self.nodes is None:
            raise CollocationError(f"Need nodes before computing weights, got {self.nodes}")

        # Instantiate the Lagrange interpolator object
        approx = LagrangeApproximation(self.nodes, weightComputation='AUTO')

        # Compute weights
        tLeft = np.ravel(self.tleft)[0]
        tRight = np.ravel(self.tright)[0]
        weights = approx.getIntegrationMatrix([(tLeft, tRight)], numQuad='FEJER')

        return np.ravel(weights)

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
        nodes += 1.0
        nodes /= 2.0
        nodes *= b - a
        nodes += a

        if self.left_is_node:
            nodes[0] = self.tleft
        if self.right_is_node:
            nodes[-1] = self.tright

        return nodes

    @property
    def _gen_Qmatrix(self):
        """
        Compute tleft-to-node integration matrix for later use in collocation formulation

        Returns:
            numpy.ndarray: matrix containing the weights for tleft to node
        """
        if self.nodes is None:
            raise CollocationError(f"Need nodes before computing weights, got {self.nodes}")
        M = self.num_nodes
        Q = np.zeros([M + 1, M + 1])

        # Instantiate the Lagrange interpolator object
        approx = LagrangeApproximation(self.nodes, weightComputation='AUTO')

        # Compute tleft-to-node integration matrix
        tLeft = np.ravel(self.tleft)[0]
        intervals = [(tLeft, tau) for tau in self.nodes]
        intQ = approx.getIntegrationMatrix(intervals, numQuad='FEJER')

        # Store into Q matrix
        Q[1:, 1:] = intQ

        return Q

    @property
    def _gen_Smatrix(self):
        """
        Compute node-to-node integration matrix for later use in collocation formulation

        Returns:
            numpy.ndarray: matrix containing the weights for node to node
        """
        M = self.num_nodes
        Q = self.Qmat
        S = np.zeros([M + 1, M + 1])

        S[1, :] = Q[1, :]
        for m in np.arange(2, M + 1):
            S[m, :] = Q[m, :] - Q[m - 1, :]

        return S

    @property
    def _gen_deltas(self):
        """
        Compute distances between the nodes

        Returns:
            numpy.ndarray: distances between the nodes
        """
        M = self.num_nodes
        delta = np.zeros(M)
        delta[0] = self.nodes[0] - self.tleft
        for m in np.arange(1, M):
            delta[m] = self.nodes[m] - self.nodes[m - 1]

        return delta

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
                intpl.splrep(self.nodes, np.roll(circ_one, i), xb=self.tleft, xe=self.tright, k=self.order, s=0.0)
            )

        weights = np.zeros(self.num_nodes)
        for i in range(self.num_nodes):
            weights[i] = intpl.splint(a, b, tcks[i])

        return weights

    @property
    def _gen_Qmatrix_spline(self):
        """
        Compute tleft-to-node integration matrix for later use in collocation formulation
        Returns:
            numpy.ndarray: matrix containing the weights for tleft to node
        """
        M = self.num_nodes
        Q = np.zeros([M + 1, M + 1])

        # for all nodes, get weights for the interval [tleft,node]
        for m in np.arange(M):
            Q[m + 1, 1:] = self._getWeights(self.tleft, self.nodes[m])

        return Q
