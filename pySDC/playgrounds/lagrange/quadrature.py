#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 14:07:30 2022

@author: cpf5546
"""
import numpy as np
import matplotlib.pyplot as plt
from time import time

from pySDC.core import NodesGenerator, LagrangeApproximation
from pySDC.implementations.collocations import Collocation

from pySDC.core.Errors import CollocationError

from scipy.integrate import quad
from scipy.interpolate import BarycentricInterpolator

class OriginCollocation(Collocation):

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
            raise CollocationError(
                "Need nodes before computing weights, got %s" % self.nodes)

        circ_one = np.zeros(self.num_nodes)
        circ_one[0] = 1.0
        tcks = []
        for i in range(self.num_nodes):
            tcks.append(BarycentricInterpolator(
                self.nodes, np.roll(circ_one, i)))

        weights = np.zeros(self.num_nodes)
        for i in range(self.num_nodes):
            weights[i] = quad(tcks[i], a, b, epsabs=1e-14)[0]

        return weights

    @property
    def _gen_Qmatrix(self):
        """
        Compute tleft-to-node integration matrix for later use in collocation
        formulation
        Returns:
            numpy.ndarray: matrix containing the weights for tleft to node
        """
        M = self.num_nodes
        Q = np.zeros([M + 1, M + 1])

        # for all nodes, get weights for the interval [tleft,node]
        for m in np.arange(M):
            Q[m + 1, 1:] = self._getWeights(self.tleft, self.nodes[m])

        return Q


def getLastPlotCol():
    return plt.gca().get_lines()[-1].get_color()


nodeTypes = ['EQUID', 'LEGENDRE']
quadTypes = ['LOBATTO', 'RADAU-LEFT', 'RADAU-RIGHT', 'GAUSS']
symbols = ['s', '>', '<', 'o']

nMax = 12
nNodes = np.arange(3, nMax + 1)

nInterTest = 10
nPolyTest = 20
baryWeightComputation = 'FAST'


QMatrixOrder = lambda n: n - 1 - (n % 2)

weightsOrder = {
    'EQUID': lambda n: n - 1 - (n % 2),
    'LEGENDRE' : {
        'LOBATTO' : lambda n: 2 * n - 3,
        'RADAU-LEFT' : lambda n: 2 * n - 2,
        'RADAU-RIGHT' : lambda n: 2 * n - 2,
        'GAUSS' : lambda n: 2 * n - 1
        }
    }


def testWeights(weights, nodes, orderFunc, tBeg, tEnd):
    deg = orderFunc(np.size(nodes))
    err = np.zeros(nPolyTest)
    for i in range(nPolyTest):
        poly_coeff = np.random.rand(deg+1)
        poly_vals  = np.polyval(poly_coeff, nodes)
        poly_int_coeff = np.polyint(poly_coeff)
        int_ex = np.polyval(poly_int_coeff, tEnd) \
            - np.polyval(poly_int_coeff, tBeg)
        int_coll = np.sum(weights * poly_vals)
        err[i] = abs(int_ex-int_coll)
    return err


def testQMatrix(QMatrix, nodes, tBeg):
    n = np.size(nodes)
    deg = QMatrixOrder(n)
    err = np.zeros((nPolyTest, n))
    for i in range(nPolyTest):
        poly_coeff = np.random.rand(deg + 1)
        poly_vals  = np.polyval(poly_coeff, nodes)
        poly_int_coeff = np.polyint(poly_coeff)
        for j in range(n):
            int_ex = np.polyval(poly_int_coeff, nodes[j])
            int_ex -= np.polyval(poly_int_coeff, tBeg)
            int_coll = np.sum(poly_vals * QMatrix[j,:])
            err[i, j] = abs(int_ex-int_coll)
    return err


def computeQuadratureErrors(nodeType, quadType, numQuad):

    errors = np.zeros((4, nMax - 2))
    errWeights = errors[:2]
    errQuad = errors[2:]

    tComp = np.zeros(nMax - 2)

    for i in range(nInterTest):

        tLeft = np.random.rand() * 0.2
        tRight = 0.8 + np.random.rand() * 0.2

        for l, n in enumerate(nNodes):

            tBeg = time()
            if numQuad == 'ORIG':
                # Use origin collocation class
                coll = OriginCollocation(n, tLeft, tRight, nodeType, quadType)
                nodes = coll.nodes
                weights = coll.weights
                QMatrix = coll.Qmat[1:, 1:]
            elif numQuad == 'NEW':
                # Use collocation class
                coll = Collocation(n, tLeft, tRight, nodeType, quadType)
                nodes = coll.nodes
                weights = coll.weights
                QMatrix = coll.Qmat[1:, 1:]
            else:
                # Generate nodes
                nodesGen = NodesGenerator(nodeType, quadType)
                nodes = nodesGen.getNodes(n)
                a = tLeft
                b = tRight
                nodes += 1
                nodes /= 2
                nodes *= b - a
                nodes += a
                # Set-up Lagrange interpolation polynomial
                approx = LagrangeApproximation(
                    nodes, weightComputation=baryWeightComputation)
                # Compute quadrature weights for the whole interval
                weights = approx.getIntegrationMatrix(
                    [[tLeft, tRight]], numQuad=numQuad)
                # Compute quadrature weights for the Q matrix
                QMatrix = approx.getIntegrationMatrix(
                    [[tLeft, tau] for tau in approx.points], numQuad=numQuad)

            tComp[l] += time() - tBeg

            # Get the corresponding accuracy order
            try:
                orderFunc = weightsOrder[nodeType]
                orderFunc(1990)
            except TypeError:
                orderFunc = orderFunc[quadType]

            # Test weights error
            err = testWeights(weights, nodes, orderFunc, tLeft, tRight)
            errWeights[0, l] += np.sum(err)
            errWeights[1, l] = max(np.max(err), errWeights[1, l])

            # Test quadrature matrix error
            err = testQMatrix(QMatrix, nodes, tLeft)
            errQuad[0, l] += np.sum(err)
            errQuad[1, l] = max(np.max(err), errQuad[1, l])

    errWeights[0] /= nPolyTest * nInterTest
    errQuad[0] /= nPolyTest * nInterTest * nNodes
    tComp /= nInterTest

    return errors, tComp


def plotQuadErrors(nodesType, numQuad, figTitle=False):

    def setFig(title, err=True):
        plt.title(title)
        plt.grid(True)
        plt.legend()
        if err:
            if numQuad == 'ORIG':
                plt.ylim(1e-17, 1e-10)
            else:
                plt.ylim(1e-17, 1e-12)
        plt.xlabel('Polynomial degree')

    plt.figure()
    maxWeigts = 0
    maxQMatrix = 0
    for qType, sym in zip(quadTypes, symbols):

        errs, tComp = computeQuadratureErrors(nodesType, qType, numQuad)

        plt.subplot(1, 3, 1)
        plt.semilogy(nNodes - 1, errs[0], sym + '-', label=qType)
        plt.semilogy(nNodes - 1, errs[1], sym + ':', c=getLastPlotCol())
        maxWeigts = max(maxWeigts, errs[1].max())
        setFig('Weights error')

        plt.subplot(1, 3, 2)
        plt.semilogy(nNodes - 1, errs[2], sym + '-', label=qType)
        plt.semilogy(nNodes - 1, errs[3], sym + ':', c=getLastPlotCol())
        maxQMatrix = max(maxQMatrix, errs[3].max())
        setFig('QMatrix error')

        plt.subplot(1, 3, 3)
        plt.semilogy(nNodes - 1, tComp, sym + '-', label=qType)
        setFig('Computation time', err=False)

    textArgs = dict(
        bbox=dict(boxstyle="round",
                  ec=(0.5, 0.5, 0.5),
                  fc=(0.8, 0.8, 0.8)))
    plt.subplot(1, 3, 1)
    plt.hlines(maxWeigts, 2, nMax, linestyles='--', colors='gray')
    plt.text(7, 1.5*maxWeigts, f'{maxWeigts:1.2e}', **textArgs)

    plt.subplot(1, 3, 2)
    plt.hlines(maxQMatrix, 2, nMax, linestyles='--', colors='gray')
    plt.text(7, 1.5*maxQMatrix, f'{maxQMatrix:1.2e}', **textArgs)

    if figTitle:
        plt.suptitle(f'node distribution : {nodesType}; '
                      f'numQuad : {numQuad}')
    plt.gcf().set_size_inches(17, 5)
    plt.tight_layout()

if __name__ == '__main__':
    nodesType = 'EQUID'
    numQuad = 'ORIG'
    plotQuadErrors(nodesType, numQuad)
