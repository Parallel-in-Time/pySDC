import logging
import numpy as np
from qmat import Q_GENERATORS

from pySDC.core.errors import CollocationError


class CollBase(object):
    """
    Generic collocation class, that contains everything to do integration over
    intervals and between nodes.
    It can be used to produce many kind of quadrature nodes from various
    distribution (awesome!).

    It is based on the two main parameters that define the nodes:

    - node_type: the node distribution used for the collocation method
    - quad_type: the type of quadrature used (inclusion of not of boundary)

    Current implementation provides the following available parameter values
    for node_type:

    - EQUID: equidistant node distribution
    - LEGENDRE: distribution from Legendre polynomials
    - CHEBY-{1,2,3,4}: distribution from Chebyshev polynomials of a given kind

    The type of quadrature can be GAUSS (only inner nodes), RADAU-LEFT
    (inclusion of the left boundary), RADAU-RIGHT (inclusion of the right
    boundary) and LOBATTO (inclusion of left and right boundary).

    All coefficients are generated using
    `qmat <https://qmat.readthedocs.io/en/latest/autoapi/qmat/qcoeff/collocation/index.html>`_.

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

    def __init__(self, num_nodes=None, tleft=0, tright=1, node_type='LEGENDRE', quad_type=None, **kwargs):
        """
        Initialization routine for a collocation object

        Args:
            num_nodes (int): number of collocation nodes
            tleft (float): left interval point
            tright (float): right interval point
        """

        if not num_nodes > 0:
            raise CollocationError('at least one quadrature node required, got %s' % num_nodes)
        if not tleft < tright:
            raise CollocationError('interval boundaries are corrupt, got %s and %s' % (tleft, tright))

        self.logger = logging.getLogger('collocation')
        try:
            self.generator = Q_GENERATORS["Collocation"](
                nNodes=num_nodes, nodeType=node_type, quadType=quad_type, tLeft=tleft, tRight=tright
            )
        except Exception as e:
            raise CollocationError(f"could not instantiate qmat generator, got error: {e}") from e

        # Set base attributes
        self.num_nodes = num_nodes
        self.tleft = tleft
        self.tright = tright
        self.node_type = node_type
        self.quad_type = quad_type
        self.left_is_node = self.quad_type in ['LOBATTO', 'RADAU-LEFT']
        self.right_is_node = self.quad_type in ['LOBATTO', 'RADAU-RIGHT']

        # Integration order
        self.order = self.generator.order

        # Compute coefficients
        self.nodes = self._getNodes = self.generator.nodes.copy()
        self.weights = self.generator.weights.copy()

        Q = np.zeros([num_nodes + 1, num_nodes + 1], dtype=float)
        Q[1:, 1:] = self.generator.Q
        self.Qmat = Q

        S = np.zeros([num_nodes + 1, num_nodes + 1], dtype=float)
        S[1:, 1:] = super(self.generator.__class__, self.generator).S
        # Note: qmat redefines the S matrix for collocation with integrals,
        # instead of differences of the Q matrix coefficients.
        # This does not passes the pySDC tests ... however the default S computation
        # in qmat uses Q matrix coefficients differences, and that's what we
        # use by using the parent property from the generator object.
        self.Smat = self._gen_Smatrix = S

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
