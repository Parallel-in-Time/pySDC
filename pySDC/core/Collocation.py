from __future__ import division
import abc
from future.utils import with_metaclass
from scipy import interpolate
from scipy.interpolate import BarycentricInterpolator
from scipy.integrate import quad
import numpy as np
import logging

from pySDC.core.Errors import CollocationError


class CollBase(with_metaclass(abc.ABCMeta)):
    """
    Abstract class for collocation

    Derived classes will contain everything to do integration over intervals and between nodes, they only need to
    provide the set of nodes, the rest is done here (awesome!)

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
        do_DG (bool): option to use DG-based Q matrix
    """

    def __init__(self, num_nodes, tleft=0, tright=1, do_DG=False):
        """
        Initialization routine for an collocation object

        Args:
            num_nodes (int): number of collocation nodes
            tleft (float): left interval point
            tright (float): right interval point
            do_DG (bool): option to use DG-based Q matrix
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

        # Dummy values for the rest
        self.nodes = None
        self.weights = None
        self.Qmat = None
        self.Smat = None
        self.delta_m = None
        self.right_is_node = None
        self.left_is_node = None
        self.do_DG = do_DG

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

    def __set_DG(self, p):

        L = np.zeros([p + 1, p + 1])
        Lagrange = [interpolate.lagrange(self.nodes, np.eye(p + 1)[i, :]) for i in range(p + 1)]

        for i in range(p + 1):
            for j in range(p + 1):
                L_Int = np.polyint(np.polymul(np.polyder(Lagrange[i]), Lagrange[j]))
                L[i, j] = np.polyval(L_Int, 1) - np.polyval(L_Int, 0)

        L[p, p] -= 1
        L_inv = np.linalg.inv(L)

        return L_inv

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
            raise CollocationError("Need nodes before computing weights, got %s" % self.nodes)

        circ_one = np.zeros(self.num_nodes)
        circ_one[0] = 1.0
        tcks = []
        for i in range(self.num_nodes):
            tcks.append(BarycentricInterpolator(self.nodes, np.roll(circ_one, i)))

        weights = np.zeros(self.num_nodes)
        for i in range(self.num_nodes):
            weights[i] = quad(tcks[i], a, b, epsabs=1e-14)[0]

        return weights

    @abc.abstractmethod
    def _getNodes(self):
        """
        Dummy method for generating the collocation nodes.
        """
        pass

    @property
    def _gen_Qmatrix(self):
        """
        Compute tleft-to-node integration matrix for later use in collocation formulation

        Returns:
            numpy.ndarray: matrix containing the weights for tleft to node
        """
        M = self.num_nodes
        Q = np.zeros([M + 1, M + 1])

        if not self.do_DG:
            # for all nodes, get weights for the interval [tleft,node]
            for m in np.arange(M):
                Q[m + 1, 1:] = self._getWeights(self.tleft, self.nodes[m])
        else:
            if not self.right_is_node:
                raise CollocationError('This will only work if the right point is a quadrature node!')
            L_inv = self.__set_DG(self.num_nodes - 1)
            Q[1:, 1:] = -L_inv.dot(np.diag(self.weights))

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
