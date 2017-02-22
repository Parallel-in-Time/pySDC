from __future__ import division
import numpy as np
import scipy.sparse as sp

from pySDC.core.Collocation import CollBase
from pySDC.core.Errors import CollocationError


class CollGaussRadau_Left(CollBase):
    """
    Implements Gauss-Radau Quadrature with left interval boundary included

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
        super(CollGaussRadau_Left, self).__init__(num_nodes, tleft, tright, do_DG)
        if num_nodes < 2:
            raise CollocationError("Number of nodes should be at least 2 for Gauss-Radau, but is %d" % num_nodes)
        self.order = 2 * self.num_nodes - 1
        self.nodes = self._getNodes
        self.left_is_node = True
        self.right_is_node = False
        self.weights = self._getWeights(tleft, tright)
        self.Qmat = self._gen_Qmatrix
        self.Smat = self._gen_Smatrix
        self.delta_m = self._gen_deltas

    @property
    def _getNodes(self):
        """
        Computes Gauss-Radau integration nodes with left point included.

        Copyright by Daniel Ruprecht (who copied this from somewhere else), 2014

        Returns:
            np.ndarray: array of Gauss-Radau nodes
        """
        M = self.num_nodes
        a = self.tleft
        b = self.tright

        alpha = 0.0
        beta = 1.0

        diag = np.zeros(M - 1)
        subdiag = np.zeros(M - 2)

        diag[0] = (beta - alpha) / (2 + alpha + beta)

        for jj in range(1, M - 1):
            diag[jj] = (beta - alpha) * (alpha + beta) / (2 * jj + 2 + alpha + beta) / (2 * jj + alpha + beta)
            num = np.sqrt(4 * jj * (jj + alpha) * (jj + beta) * (jj + alpha + beta))
            denom = np.sqrt((2 * jj - 1 + alpha + beta) * (2 * jj + alpha + beta) ** 2 * (2 * jj + 1 + alpha + beta))
            subdiag[jj - 1] = num / denom

        subdiag1 = np.zeros(M - 1)
        subdiag2 = np.zeros(M - 1)
        subdiag1[0:-1] = subdiag
        subdiag2[1:] = subdiag

        Mat = sp.spdiags(data=[subdiag1, diag, subdiag2], diags=[-1, 0, 1], m=M - 1, n=M - 1).todense()

        x = np.sort(np.linalg.eigvals(Mat))

        nodes = np.concatenate(([-1.0], x))

        nodes = (a * (1 - nodes) + b * (1 + nodes)) / 2

        self.logger.warning('GaussRadau_Left not fully tested, use at own risk!')

        return nodes
