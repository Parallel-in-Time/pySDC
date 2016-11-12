from __future__ import division
import numpy as np

from pySDC.core.Collocation import CollBase
from pySDC.core.Errors import CollocationError


class CollGaussLegendre(CollBase):
    """
    Implements Gauss-Legendre Quadrature

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
        super(CollGaussLegendre, self).__init__(num_nodes, tleft, tright)
        if num_nodes < 1:
            raise CollocationError("Number of nodes should be at least 1 for Gauss-Legendre, but is %d" % num_nodes)
        self.order = 2 * self.num_nodes
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
        Computes nodes for the Gauss-Legendre quadrature

        Python version by Dieter Moser, 2014

        Returns:
            np.ndarray: array of Gauss-Legendre nodes
        """
        M = self.num_nodes
        a = self.tleft
        b = self.tright

        # Building the companion matrix comp_mat with det(nodes*I-comp_mat)=P_n(nodes), where P_n is the
        # Legendre polynomial under consideration. comp_mat will be constructed in such a way that it is symmetric.
        linspace = np.linspace(1, M - 1, M - 1)
        v = [linspace[i] / np.sqrt(4.0 * linspace[i] ** 2 - 1.0) for i in range(len(linspace))]
        comp_mat = np.diag(v, 1) + np.diag(v, -1)

        # Determining the abscissas (nodes) - since det(nodesI-comp_mat)=P_n(nodes), the abscissas are the roots
        # of the characteristic polynomial, i.e. the eigenvalues of comp_mat
        [eig_vals, _] = np.linalg.eig(comp_mat)
        indizes = np.argsort(eig_vals)
        nodes = eig_vals[indizes]

        # take real part and shift from [-1,1] to [a,b]
        nodes = nodes.real
        nodes = (a * (1 - nodes) + b * (1 + nodes)) / 2

        return nodes
