from __future__ import division
import numpy as np

from pySDC.implementations.collocation_classes.equidistant import Equidistant
from pySDC.core.Errors import CollocationError, ParameterError

from scipy.integrate import quad
from scipy.interpolate import BarycentricInterpolator


class MyBarycentricInterpolator(BarycentricInterpolator):
    """
    Overwrite BarycentricInterolator to inject custom weights
    """
    def __init__(self, xi, yi=None, weights=None, axis=0):
        super(MyBarycentricInterpolator, self).__init__(xi, yi, axis)
        self.wi = weights


class Equidistant_RDC(Equidistant):
    """
    Implements equidistant nodes with blended barycentric interpolation

    Attributes:
        fh_weights: blended FH weights for barycentric interpolation
    """

    def __init__(self, num_nodes, tleft, tright):
        """
        Initialization

        Args:
            num_nodes: number of nodes
            tleft (float): left interval boundary (usually 0)
            tright (float): right interval boundary (usually 1)
        """

        if type(num_nodes) is int:
            max_d = 15
            nnodes = num_nodes
        else:
            if type(num_nodes) is not tuple:
                raise ParameterError('Expecting int or tuple for num_nodes parameter, got %s' %type(num_nodes))
            if len(num_nodes) != 2:
                raise ParameterError('Expecting 1 or 2 arguments for num_nodes, got %s' % num_nodes)
            if type(num_nodes[0]) is not int:
                raise ParameterError('Expecting int type for first num_nodes argument, got %s' % type(num_nodes[0]))
            if type(num_nodes[1]) is not int:
                raise ParameterError('Expecting int type for second num_nodes argument, got %s' % type(num_nodes[1]))
            max_d = num_nodes[1]
            nnodes = num_nodes[0]

        if nnodes < 1:
            raise CollocationError("Number of nodes should be at least 1 for equidistant, but is %d" % num_nodes)

        super(Equidistant, self).__init__(nnodes, tleft, tright)

        self.order = self.num_nodes
        self.nodes = self._getNodes

        d = min(self.num_nodes - 1, max_d)
        self.fh_weights = self._getFHWeights(d)
        self.weights = self._getWeights(tleft, tright)

        self.Qmat = self._gen_Qmatrix
        self.Smat = self._gen_Smatrix
        self.delta_m = self._gen_deltas
        self.left_is_node = True
        self.right_is_node = True

    @property
    def _getNodes(self):
        """
        Computes integration nodes with both boundaries included

        Returns:
            np.ndarray: array of equidistant nodes
        """
        M = self.num_nodes
        a = self.tleft
        b = self.tright
        nodes = np.linspace(a, b, M)
        return nodes

    def _getFHWeights(self, d):
        """
        Computes blended FH weights for barycentric interpolation

        This method is ported from Georges Klein's matlab function

        Args:
            d (int): blending parameter

        Returns:
            numpy.ndarray: weights
        """

        n = self.num_nodes - 1
        w = np.zeros(n + 1)

        for k in range(0, n + 1):
            ji = max(k - d, 0)
            jf = min(k, n - d)
            sumcoeff = []
            for i in range(ji, jf + 1):
                prodterm = []
                for j in range(i, i + d + 1):
                    if j == k:
                        prodterm.append(1)
                    else:
                        prodterm.append(self.nodes[k] - self.nodes[j])
                product = 1.0 / np.prod(prodterm)
                sumcoeff.append((-1) ** (i - 1) * product)
            y = sorted(sumcoeff, key=abs)
            w[k] = np.sum(y)

        return w

    def _getWeights(self, a, b):
        """
        Computes weights using custom barycentric interpolation

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
            # This is where the custom BarycentricInterpolator is called
            tcks.append(MyBarycentricInterpolator(self.nodes, np.roll(circ_one, i), self.fh_weights))

        weights = np.zeros(self.num_nodes)
        for i in range(self.num_nodes):
            weights[i] = quad(tcks[i], a, b, epsabs=1e-14)[0]

        return weights
