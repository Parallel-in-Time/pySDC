from pySDC.core.Collocation import CollBase
from pySDC.core.Nodes import NodesGenerator

import numpy as np
import scipy.interpolate as intpl


class Collocation(CollBase):
    """
    Generic collocation class alowing to produce many kind of quadrature nodes
    from various distribution.
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
    """

    def __init__(self, num_nodes, tleft, tright,
                 node_type='LEGENDRE', quad_type='LOBATTO', useSpline=False):
        """
        Initialization

        Args:
            num_nodes (int): number of nodes
            tleft (float): left interval boundary (usually 0)
            tright (float): right interval boundary (usually 1)
            node_type (str): node distribution to use (default LEGENDRE)
            quad_type (str): quadrature type to use (default LOBATTO)
            useSpline (bool): wether or not use spline interpolation to compute
                weights (default False)
        """
        # Base constructor
        super(Collocation, self).__init__(num_nodes, tleft, tright)
        # Instanciate attributes
        self.nodeGenerator = NodesGenerator(node_type, quad_type)
        if useSpline:
            self._getWeights = self._getWeights_spline
            # We need: 1<=order<=5 and order < num_nodes
            self.order = min(num_nodes - 1, 3)
        elif node_type == 'EQUID':
            self.order = num_nodes
        else:
            if quad_type == 'GAUSS':
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
        self.left_is_node = quad_type in ['LOBATTO', 'RADAU-LEFT']
        self.right_is_node = quad_type in ['LOBATTO', 'RADAU-RIGHT']

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
        nodes *= b - a
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
