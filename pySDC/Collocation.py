from __future__ import division
from abc import ABCMeta, abstractmethod
from scipy.interpolate import BarycentricInterpolator
from scipy.integrate import quad
import numpy as np

class CollBase(object):
    """
    Abstract class for collocation
    -> derived classes will contain everything to do integration over intervals and between nodes
    -> abstract class contains already Gauss-Legendre collocation to compute weights for arbitrary nodes
    -> child class only needs to implement the set of nodes, the rest is done here
    """

    def __init__(self, num_nodes, tleft, tright):
        """
        Initialization routine for an collocation object
        ------
        Input:
        :param num_nodes: specify number of collocation nodes
        :param tleft: left interval boundary
        :param tright: right interval boundary
        """
        # Set number of nodes, left and right interval boundaries
        assert num_nodes > 0, 'At least one quadrature node required, got %d' % num_nodes
        assert tleft < tright, 'Interval boundaries are corrupt, got %f and %f' % (tleft,tright)
        self.num_nodes = num_nodes
        self.tleft = tleft
        self.tright = tright
        # Set dummy nodes and weights
        self.nodes = None
        self.weights = None
        self.Qmat = None
        self.Smat = None
        self.delta_m = None


    @staticmethod
    def evaluate(weights, data):
        """
        :param weights: integration weights
        :param data: f(x) to be integrated
        :return: integral over f(x), where the boundaries are implicitly given by the definition of the weights
        """
        assert np.size(weights) == np.size(data), \
            "Input size does not match number of weights, but is %d" % np.size(data)
        return np.dot(weights, data)

    def _getWeights(self, a, b):
        """
        Evaluate weights using barycentric interpolation
        :param a: left interval boundary
        :param b: right interval boundary
        :return: weights of the collocation formula given by the nodes
        """
        assert self.num_nodes > 0, "Need number of nodes before computing weights, got %d" % self.num_nodes
        assert self.nodes is not None, "Need nodes before computing weights, got %d" % self.nodes
        circ_one = np.zeros(self.num_nodes)
        circ_one[0] = 1.0
        tcks = []
        for i in range(self.num_nodes):
            tcks.append(BarycentricInterpolator(self.nodes, np.roll(circ_one, i)))

        weights = np.zeros(self.num_nodes)
        for i in range(self.num_nodes):
            weights[i] = quad(tcks[i], a, b, epsabs=1e-14)[0]

        return weights

    @abstractmethod
    def _getNodes(self):
        """
        Dummy method for generating the collocation nodes.
        Will be overridden by child classes
        """
        pass

    @property
    def _gen_Qmatrix(self):
        """
        Compute tleft-to-node integration matrix for later use in collocation formulation
        :return: Q matrix
        """
        M = self.num_nodes
        Q = np.zeros([M+1, M+1])

        # for all nodes, get weights for the interval [tleft,node]
        for m in np.arange(M):
            Q[m+1, 1:] = self._getWeights(self.tleft, self.nodes[m])

        return Q

    @property
    def _gen_Smatrix(self):
        """
        Compute node-to-node inetgration matrix for later use in collocation formulation
        :return: S matrix
        """
        M = self.num_nodes
        Q = self.Qmat
        S = np.zeros([M+1, M+1])

        S[1, :] = Q[1, :]
        for m in np.arange(2, M+1):
            S[m, :] = Q[m, :] - Q[m - 1, :]

        return S

    @property
    def _gen_deltas(self):

        M = self.num_nodes
        delta = np.zeros(M)
        delta[0] = self.nodes[0] - self.tleft
        for m in np.arange(1,M):
            delta[m] = self.nodes[m] - self.nodes[m-1]

        return delta
