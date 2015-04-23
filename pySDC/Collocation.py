from __future__ import division
from abc import ABCMeta, abstractmethod
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
        self.Qdmat = None

    @staticmethod
    def _GaussLegendre(M, a, b):

        """
        % Copyright (c) 2009, Greg von Winckel
        % All rights reserved.
        %
        % Redistribution and use in source and binary forms, with or without
        % modification, are permitted provided that the following conditions are
        % met:
        %
        %     * Redistributions of source code must retain the above copyright
        %       notice, this list of conditions and the following disclaimer.
        %     * Redistributions in binary form must reproduce the above copyright
        %       notice, this list of conditions and the following disclaimer in
        %       the documentation and/or other materials provided with the distribution
        %
        % THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
        % AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
        % IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
        % ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
        % LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
        % CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
        % SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
        % INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
        % CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
        % ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
        % POSSIBILITY OF SUCH DAMAGE.

        % lgwt.m
        %
        % This script is for computing definite integrals using Legendre-Gauss
        % Quadrature. Computes the Legendre-Gauss nodes and weights  on an interval
        % [a,b] with truncation order M

        :param M: number of collocation nodes
        :param a: left interval boundary, may differ from tleft
        :param b: right interval boundary, may differ from tright
        :return: nodes and weights according to Gauss-Legendre quadrature
        """
        assert a <= b, 'Interval boundaries are corrupt, got %f and %f' % (a,b)

        M = M - 1
        M1 = M + 1
        M2 = M + 2

        xu = np.linspace(-1, 1, M1)
        # Initial guess
        y = np.cos((2 * np.arange(0, M + 1) + 1) * np.pi / (2 * M + 2)) + 0.27/M1 * np.sin(np.pi * xu * M / M2)
        # Legendre-Gauss Vandermonde Matrix
        L = np.zeros([M1, M2])
        # Derivative pf LG-VM (need only one vector a time)
        Lp = np.zeros(M2)

        # Compute the zeros of the N+1 Legendre Polynomial using the recursion relation and the Newton-Raphson method
        y0 = 2
        # Iterate until new points are uniformly within epsilon of old points
        while np.linalg.norm(y - y0, np.inf) > np.finfo(float).eps:
            L[:, 0] = 1
            L[:, 1] = y
            for k in np.arange(2, M1 + 1):
                L[:, k] = ((2 * k - 1) * y * L[:, k - 1] - (k - 1) * L[:, k - 2]) / k
            Lp = M2 * (L[:, M1 - 1] - y * L[:, M2 - 1]) / (1 - y ** 2)
            y0 = y
            y = y0 - L[:, M2 - 1] / Lp

        # Linear map from[-1,1] to [a,b]
        nodes = (a * (1 - y) + b * (1 + y)) / 2
        # Compute the weights
        weights = (b - a) / ((1 - y ** 2) * Lp ** 2) * (M2 / M1) ** 2

        # Reverse the order (small nodes first)
        nodes = nodes[::-1]
        weights = weights[::-1]

        return nodes, weights

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
        Copyright (c) 2014, Daniel Ruprecht
        All rights reserved.
        For a general set of collocation nodes, the corresponding weights can be retrieved by computing the integrals
        int_a^b over the corresponding Lagrange polynomials. This is not very efficient, though.
        :param a: left interval boundary
        :param b: right interval boundary
        :return: weights of the collocation formula given by the nodes
        """
        assert a <= b, 'Interval boundaries are corrupt, got %f and %f' % (a,b)
        M = self.num_nodes
        weights = np.zeros(M)

        # Define temporary integration method using built-in Gauss-Legendre
        # -> will need this to compute the integral from a to b over the Lagrangian polynomials
        [nodes_m, weights_m] = self._GaussLegendre(np.ceil(M / 2), a, b)

        # for each node, build Lagrangian polynomial in Newton base, evaluate at temp. integration nodes and integrate
        for j in np.arange(M):
            coeff = np.zeros(M)
            coeff[j] = 1.0
            poly = self._poly_newton(coeff)
            eval_pj = self._evaluate_horner(nodes_m,poly)
            weights[j] = self.evaluate(weights_m, eval_pj)

        return weights

    def _poly_newton(self, coeff):
        """
        Copyright (c) 2014, Daniel Ruprecht
        All rights reserved.
        Computes Lagrange polynomial in Newton representation
        :param coeff: coefficients of Lagrange polynomial (choose particular node)
        :return: coefficients of polynomial in newton representation
        """
        n = self.num_nodes
        D = np.zeros([n, n])
        D[:,0] = coeff
        for j in np.arange(2, n+1):
            for k in np.arange(j, n+1):
                D[k-1, j-1] = (D[k-1, j-2]-D[k-2, j-2])/(self.nodes[k-1]-self.nodes[k-1-(j-1)])

        return np.diag(D)


    def _evaluate_horner(self, xi, coeff):
        """
        Copyright (c) 2014, Daniel Ruprecht
        All rights reserved.
        Evaluates polynomial using Horner's scheme
        :param xi: points to evaluate at
        :param coeff: coefficients of the polynomial
        :return: evaluation of the polynomial at xi
        """
        M = self.num_nodes
        fyi = coeff[M-1]
        for i in np.arange(1, M):
            fyi = coeff[M-1-i] + (xi - self.nodes[M-1-i])*fyi

        return fyi

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

    @property
    def _gen_QDmatrix(self):
        """
        Depending on how the nodes are distributed we construct the associated deltas and  from there
        the resulting Q_delta matrix which is needed in the matrix formulation of LinearPFASST
        :return: Q_delta matrix
        """

        def q_delta(tau):
            n = tau.shape[0]
            Q_delta = np.zeros((n, n))
            i = 0
            for t in tau:
                Q_delta[i:, i] = np.ones(n-i)*t
                i += 1
            return Q_delta

        if self.tleft == self.nodes[0] and self.tright == self.nodes[-1]:
            tau = np.concatenate([np.zeros(1), self.nodes[1:]-self.nodes[:-1]])
        elif self.tleft != self.nodes[0] and self.tright != self.nodes[-1]:
            tau = np.hstack([np.asarray([0.0, self.nodes[0]-self.tleft]), self.nodes[1:]-self.nodes[:-1]])
        elif self.tleft == self.nodes[0] and self.tright != self.nodes[-1]:
            pass
        else:
            pass

        return q_delta(tau)


