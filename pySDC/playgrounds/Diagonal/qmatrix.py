#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 10:17:08 2022

@author: cpf5546
"""
import numpy as np
from scipy.linalg import lu

from pySDC.core.Nodes import NodesGenerator
from pySDC.core.Lagrange import LagrangeApproximation

# Storage for diagonaly optimized QDelta matrices
OPT_COEFFS = {
    "QmQd": {
        2: {'GAUSS':
                [(0.105662, 0.394338),
                 (0.394338, 0.105662)],
            'RADAU-LEFT':
                [(0.0, 0.333333)],
            'RADAU-RIGHT':
                [(0.166667, 0.5),
                 (0.666667, 0.0)],
            'LOBATTO':
                [(0.0, 0.5)]
            },
        3: {'GAUSS':
                [(0.037571, 0.16667, 0.29577),
                 (0.156407, 0.076528, 0.267066),
                 (0.267065, 0.076528, 0.156407),
                 (0.295766, 0.166666, 0.037567)],
            'RADAU-RIGHT':
                [(0.051684, 0.214984, 0.333334), # Winner for advection
                 (0.233475, 0.080905, 0.285619),
                 (0.390077, 0.094537, 0.115385),
                 (0.422474, 0.177525, 0.0)],
            'LOBATTO':
                [(0.0, 0.166667, 0.333333),
                 (0.0, 0.5, 0.0)],
            },
        5: {'RADAU-RIGHT':
                [(0.193913, 0.141717, 0.071975, 0.018731, 0.119556),
                 (0.205563, 0.143134, 0.036388, 0.073742, 0.10488),
                 (0.176822, 0.124251, 0.031575, 0.084012, 0.142621)],
            },

        },
    "Speck": {
        2: {'GAUSS':
                [(0.166667, 0.5),
             	 (0.5, 0.166667)],
            'RADAU-LEFT':
                [(0.0, 0.333333)],
            'RADAU-RIGHT':
                [(0.258418, 0.644949),
                 (1.074915, 0.155051)],
            'LOBATTO':
                [(0.0, 0.5)]
            },
        3: {'GAUSS':
                [(0.07672, 0.258752, 0.419774),
                 (0.214643, 0.114312, 0.339631),
                 (0.339637, 0.114314, 0.214647),
                 (0.419779, 0.258755, 0.076721)],
            'RADAU-RIGHT':
                [(0.10405, 0.332812, 0.48129),
                 (0.320383, 0.139967, 0.371668), # Winner for advection
                 (0.558747, 0.136536, 0.218466),
                 (0.747625, 0.404063, 0.055172)],
            'LOBATTO':
                [(0.0, 0.211325, 0.394338),
                 (0.0, 0.788675, 0.105662)]
            },
        }
    }

# Coefficient allowing A-stability with prolongation=True
WEIRD_COEFFS = {
    'GAUSS':
        {2: (0.5, 0.5)},
    'RADAU-RIGHT':
        {2: (0.5, 0.5)},
    'RADAU-LEFT':
        {3: (0.0, 0.5, 0.5)},
    'LOBATTO':
        {3: (0.0, 0.5, 0.5)}}


def genQDelta(nodes, sweepType, Q):
    """
    Generate QDelta matrix for a given node distribution

    Parameters
    ----------
    nodes : array (M,)
        quadrature nodes, scaled to [0, 1]
    sweepType : str
        Type of sweep, that defines QDelta. Can be selected from :
    - BE : Backward Euler sweep (first order)
    - FE : Forward Euler sweep (first order)
    - LU : uses the LU trick
    - TRAP : sweep based on Trapezoidal rule (second order)
    - EXACT : don't bother and just use Q
    - PIC : Picard iteration => zeros coefficient (cannot be used for initSweep)
    - OPT-[...] : Diagonaly precomputed coefficients, for which one has to
      provide different parameters. For instance, [...]='QmQd-2' uses the
      diagonal coefficients using the optimization method QmQd with the index 2
      solution (index starts at 0 !). Quadtype and number of nodes are
      determined automatically from the Q matrix.
    - WEIRD-[...] : diagonal coefficient allowing A-stability with collocation
      update (forceProl=True).
    Q : array (M,M)
        Q matrix associated to the node distribution
        (used only when sweepType in [LU, EXACT, OPT-[...], WEIRD]).

    Returns
    -------
    QDelta : array (M,M)
        The generated QDelta matrix.
    dtau : float
        Correction coefficient for time integration with QDelta
    """
    # Generate deltas
    deltas = np.copy(nodes)
    deltas[1:] = np.ediff1d(nodes)

    # Extract informations from Q matrix
    M = deltas.size
    if sweepType.startswith('OPT') or sweepType == 'WEIRD':
        leftIsNode = np.allclose(Q[0], 0)
        rightIsNode = np.isclose(Q[-1].sum(), 1)
        quadType = 'LOBATTO' if (leftIsNode and rightIsNode) else \
            'RADAU-LEFT' if leftIsNode else \
            'RADAU-RIGHT' if rightIsNode else \
            'GAUSS'

    # Compute QDelta
    QDelta = np.zeros((M, M))
    dtau = 0.0
    if sweepType in ['BE', 'FE']:
        offset = 1 if sweepType == 'FE' else 0
        for i in range(offset, M):
            QDelta[i:, :M-i] += np.diag(deltas[offset:M-i+offset])
        if sweepType == 'FE':
            dtau = deltas[0]
    elif sweepType == 'TRAP':
        for i in range(0, M):
            QDelta[i:, :M-i] += np.diag(deltas[:M-i])
        for i in range(1, M):
            QDelta[i:, :M-i] += np.diag(deltas[1:M-i+1])
        QDelta /= 2.0
        dtau = deltas[0]/2.0
    elif sweepType == 'LU':
        QT = Q.T.copy()
        [_, _, U] = lu(QT, overwrite_a=True)
        QDelta = U.T
    elif sweepType == 'EXACT':
        QDelta = np.copy(Q)
    elif sweepType == 'PIC':
        QDelta = np.zeros(Q.shape)
    elif sweepType.startswith('OPT'):
        try:
            oType, idx = sweepType[4:].split('-')
        except ValueError:
            raise ValueError(f'missing parameter(s) in sweepType={sweepType}')
        M, idx = int(M), int(idx)
        try:
            coeffs = OPT_COEFFS[oType][M][quadType][idx]
            QDelta[:] = np.diag(coeffs)
        except (KeyError, IndexError):
            raise ValueError('no OPT diagonal coefficients for '
                             f'{oType}-{M}-{quadType}-{idx}')
    elif sweepType == 'BEpar':
        QDelta[:] = np.diag(nodes)
    elif sweepType == 'WEIRD':
        try:
            coeffs = WEIRD_COEFFS[quadType][M]
            QDelta[:] = np.diag(coeffs)
        except (KeyError, IndexError):
            raise ValueError('no WEIRD diagonal coefficients for '
                             f'{M}-{quadType} nodes')
    else:
        raise NotImplementedError(f'sweepType={sweepType}')
    return QDelta, dtau


def genCollocation(M, distr, quadType):
    """
    Generate the nodes, weights and Q matrix for a given collocation method

    Parameters
    ----------
    M : int
        Number of quadrature nodes.
    distr : str
        Node distribution. Can be selected from :
    - LEGENDRE : nodes from the Legendre polynomials
    - EQUID : equidistant nodes distribution
    - CHEBY-{1,2,3,4} : nodes from the Chebyshev polynomial (1st to 4th kind)
    quadType : str
        Quadrature type. Can be selected from :
    - GAUSS : do not include the boundary points in the nodes
    - RADAU-LEFT : include left boundary points in the nodes
    - RADAU-RIGHT : include right boundary points in the nodes
    - LOBATTO : include both boundary points in the nodes

    Returns
    -------
    nodes : array (M,)
        quadrature nodes, scaled to [0, 1]
    weights : array (M,)
        quadrature weights associated to the nodes
    Q : array (M,M)
        normalized Q matrix of the collocation problem
    """

    # Generate nodes between [0, 1]
    nodes = NodesGenerator(node_type=distr, quad_type=quadType).getNodes(M)
    nodes += 1
    nodes /= 2
    np.round(nodes, 14, out=nodes)

    # Compute Q and weights
    approx = LagrangeApproximation(nodes)
    Q = approx.getIntegrationMatrix([(0, tau) for tau in nodes])
    weights = approx.getIntegrationMatrix([(0, 1)]).ravel()

    return nodes, weights, Q
