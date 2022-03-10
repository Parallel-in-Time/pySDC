#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 14:07:30 2022

@author: cpf5546
"""
import numpy as np
import matplotlib.pyplot as plt

from pySDC.core import NodesGenerator, LagrangeApproximation
from pySDC.implementations.collocations import Collocation

nodeTypes = ['EQUID', 'LEGENDRE']
quadTypes = ['LOBATTO', 'RADAU-LEFT', 'RADAU-RIGHT', 'GAUSS']
symbols = ['s', '>', '<', 'o']

def getLastPlotCol():
    return plt.gca().get_lines()[-1].get_color()

nMax = 12
nVals = np.arange(3, nMax+1)
tBeg = -1
tEnd = 1
nExp = 20


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


def testWeights(weights, nodes, orderFunc):
    deg = orderFunc(np.size(nodes))
    err = np.zeros(nExp)
    for i in range(nExp):
        poly_coeff = np.random.rand(deg+1)
        poly_vals  = np.polyval(poly_coeff, nodes)
        poly_int_coeff = np.polyint(poly_coeff)
        int_ex = np.polyval(poly_int_coeff, tEnd) \
            - np.polyval(poly_int_coeff, tBeg)
        int_coll = np.sum(weights * poly_vals)
        err[i] = abs(int_ex-int_coll)
    return err


def testQMatrix(QMatrix, nodes):
    n = np.size(nodes)
    deg = QMatrixOrder(n)
    err = np.zeros((nExp, n))
    for i in range(nExp):
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

    nodesGen = NodesGenerator(nodeType, quadType)

    errors = np.ones((4, nMax - 2))
    errWeights = errors[:2]
    errQuad = errors[2:]

    for l, n in enumerate(nVals):

        # Generate nodes
        nodes = nodesGen.getNodes(n)

        if numQuad == 'ORIG':
            # Use collocation class
            coll = Collocation(n, tBeg, tEnd, nodeType, quadType)
            weights = coll.weights
            QMatrix = coll.Qmat[1:, 1:]
        else:
            # Set-up Lagrange interpolation polynomial
            approx = LagrangeApproximation(nodes, weightComputation='FAST')
            # Compute quadrature weights for the whole interval
            weights = approx.getIntegrationMatrix(
                [[tBeg, tEnd]], numQuad=numQuad)
            # Compute quadrature weights for the Q matrix
            QMatrix = approx.getIntegrationMatrix(
                [[tBeg, tau] for tau in approx.points], numQuad=numQuad)



        # Test weights according to the corresponding accuracy order
        try:
            orderFunc = weightsOrder[nodeType]
            orderFunc(1990)
        except TypeError:
            orderFunc = orderFunc[quadType]
        err = testWeights(weights, nodes, orderFunc)
        errWeights[0, l] = np.mean(err)
        errWeights[1, l] = np.max(err)

        err = testQMatrix(QMatrix, nodes)
        errQuad[0, l] = np.mean(err)
        errQuad[1, l] = np.max(err)

    return errors


def plotQuadErrors(nodesType, numQuad, figTitle=False):

    def setFig(title):
        plt.title(title)
        plt.grid(True)
        plt.legend()
        plt.ylim(1e-17, 1e-12)
        plt.xlabel('Polynomial degree')

    plt.figure()
    maxWeigts = 0
    maxQMatrix = 0
    for qType, sym in zip(quadTypes, symbols):
        errs = computeQuadratureErrors(nodesType, qType, numQuad)
        plt.subplot(1, 2, 1)
        plt.semilogy(nVals - 1, errs[0], sym + '-', label=qType)
        plt.semilogy(nVals - 1, errs[1], sym + ':', c=getLastPlotCol())
        maxWeigts = max(maxWeigts, errs[1].max())
        setFig('Weights error')
        plt.subplot(1, 2, 2)
        plt.semilogy(nVals - 1, errs[2], sym + '-', label=qType)
        plt.semilogy(nVals - 1, errs[3], sym + ':', c=getLastPlotCol())
        maxQMatrix = max(maxQMatrix, errs[3].max())
        setFig('QMatrix error')


    textArgs = dict(
        bbox=dict(boxstyle="round",
                  ec=(0.5, 0.5, 0.5),
                  fc=(0.8, 0.8, 0.8)))
    plt.subplot(1, 2, 1)
    plt.hlines(maxWeigts, 2, nMax, linestyles='--', colors='gray')
    plt.text(7, 1.5*maxWeigts, f'{maxWeigts:1.2e}', **textArgs)

    plt.subplot(1, 2, 2)
    plt.hlines(maxQMatrix, 2, nMax, linestyles='--', colors='gray')
    plt.text(7, 1.5*maxQMatrix, f'{maxQMatrix:1.2e}', **textArgs)

    if figTitle:
        plt.suptitle(f'node distribution : {nodesType}; '
                      f'numQuad : {numQuad}')
    plt.gcf().set_size_inches(12, 5)
    plt.tight_layout()

if __name__ == '__main__':
    nodesType = 'EQUID'
    numQuad = 'FEJER'
    plotQuadErrors(nodesType, numQuad)
