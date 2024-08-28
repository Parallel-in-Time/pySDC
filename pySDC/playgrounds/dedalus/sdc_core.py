#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Core class for IMEX SDC (independent of any application code).
Set the main parameters, compute coefficients, ...
"""
import numpy as np

from qmat.qcoeff.collocation import Collocation
from qmat import genQDeltaCoeffs

# -----------------------------------------------------------------------------
# Main SDC parameters
# -----------------------------------------------------------------------------
DEFAULT = {
    'nNodes': 4,
    'nSweeps': 3,
    'nodeDistr': 'LEGENDRE',
    'quadType': 'RADAU-RIGHT',
    'implSweep': 'BE',
    'explSweep': 'FE',
    'initSweep': 'COPY',
    'initSweepQDelta': 'BE',
    'forceProl': False,
    }

PARAMS = {
    ('nNodes', '-sM', '--sdcNNodes'):
        dict(help='number of nodes for SDC', type=int,
             default=DEFAULT['nNodes']),
    ('quadType', '-sqt', '--sdcQuadType'):
        dict(help='quadrature type for SDC',
             default=DEFAULT['quadType']),
    ('nodeDistr', '-snd', '--sdcNodeDistr'):
        dict(help='node distribution for SDC',
             default=DEFAULT['nodeDistr']),
    ('nSweeps', '-sK', '--sdcNSweeps'):
        dict(help='number of sweeps for SDC', type=int,
             default=DEFAULT['nSweeps']),
    ('initSweep', '-si', '--sdcInitSweep'):
        dict(help='initial sweep to get initialized nodes values',
             default=DEFAULT['initSweep']),
    ('initSweepQDelta', '-siq', '--sdcInitSweepQDelta'):
        dict(help='QDelta matrix used with initSweep=QDELTA',
             default=DEFAULT['initSweepQDelta']),
    ('implSweep', '-sis', '--sdcImplSweep'):
        dict(help='type of QDelta matrix for implicit sweep',
             default=DEFAULT['implSweep']),
    ('explSweep', '-ses', '--sdcExplSweep'):
        dict(help='type of QDelta matrix for explicit sweep',
             default=DEFAULT['explSweep']),
    ('forceProl', '-sfp', '--sdcForceProl'):
        dict(help='if specified force the prolongation stage '
             '(ignored for quadType=GAUSS or RADAU-LEFT)',
             action='store_true')
    }

# Printing function
def sdcInfos(nNodes, quadType, nodeDistr, nSweeps,
             implSweep, explSweep, initSweep, forceProl,
             **kwargs):
    return f"""
-- nNodes : {nNodes}
-- quadType : {quadType}
-- nodeDistr : {nodeDistr}
-- nSweeps : {nSweeps}
-- implSweep : {implSweep}
-- explSweep : {explSweep}
-- initSweep : {initSweep}
-- forceProl : {forceProl}
""".strip()

# -----------------------------------------------------------------------------
# Base class implementation
# -----------------------------------------------------------------------------
class IMEXSDCCore(object):

    # Initialize parameters with default values
    nSweeps:int = DEFAULT['nSweeps']
    nodeType:str = DEFAULT['nodeDistr']
    quadType:str = DEFAULT['quadType']
    implSweep = DEFAULT['implSweep']
    explSweep = DEFAULT['explSweep']
    initSweep = DEFAULT['initSweep']
    initSweepQDelta = DEFAULT['initSweepQDelta']
    forceProl = DEFAULT['forceProl']

    # Collocation method attributes
    coll = Collocation(
        nNodes=DEFAULT['nNodes'], nodeType=nodeType, quadType=quadType)
    nodes, weights, Q = coll.genCoeffs()
    # IMEX SDC attributes, QDelta matrices are 3D with shape (K, M, M)
    QDeltaI, dtauI = genQDeltaCoeffs(
        implSweep, dTau=True, nSweeps=nSweeps,
        Q=Q, nodes=nodes, nNodes=DEFAULT['nNodes'],
        nodeType=nodeType, quadType=quadType)
    QDeltaE, dtauE = genQDeltaCoeffs(
        explSweep, dTau=True, nSweeps=nSweeps,
        Q=Q, nodes=nodes, nNodes=DEFAULT['nNodes'],
        nodeType=nodeType, quadType=quadType)
    QDelta0 = genQDeltaCoeffs(
        initSweepQDelta,
        Q=Q, nodes=nodes, nNodes=DEFAULT['nNodes'],
        nodeType=nodeType, quadType=quadType)

    diagonal = np.all(np.diag(np.diag(qD)) == qD for qD in QDeltaI)
    diagonal *= np.all(np.diag(np.diag(qD)) == 0 for qD in QDeltaE)
    if initSweep == "QDelta":
        diagonal *=  np.all(np.diag(np.diag(QDelta0)) == QDelta0)

    # Default attributes, used later as instance attributes
    # => should be defined in inherited class
    dt = None
    axpy = None

    @classmethod
    def setParameters(cls, nNodes=None, nodeType=None, quadType=None,
                      implSweep=None, explSweep=None, initSweep=None,
                      initSweepQDelta=None, nSweeps=None, forceProl=None):

        # Get non-changing parameters
        keepNNodes = nNodes is None
        keepNodeDistr = nodeType is None
        keepQuadType = quadType is None
        keepImplSweep = implSweep is None
        keepExplSweep = explSweep is None
        keepNSweeps = nSweeps is None
        keepInitSweepQDelta = initSweepQDelta is None

        # Set parameter values
        nNodes = len(cls.nodes) if keepNNodes else nNodes
        nodeType = cls.nodeType if keepNodeDistr else nodeType
        quadType = cls.quadType if keepQuadType else quadType
        implSweep = cls.implSweep if keepImplSweep else implSweep
        explSweep = cls.explSweep if keepExplSweep else explSweep
        nSweeps = cls.nSweeps if keepNSweeps else nSweeps
        initSweepQDelta = cls.initSweepQDelta if keepInitSweepQDelta else initSweepQDelta

        # Determine updated parts
        updateNode = (not keepNNodes) or (not keepNodeDistr) or (not keepQuadType)
        updateQDeltaI = (not keepImplSweep) or updateNode or (not keepNSweeps)
        updateQDeltaE = (not keepExplSweep) or updateNode or (not keepNSweeps)
        updateQDelta0 = (not keepInitSweepQDelta) or updateNode

        # Update parameters
        if updateNode:
            cls.coll = Collocation(
                nNodes=DEFAULT['nNodes'], nodeType=nodeType, quadType=quadType)
            cls.nodes, cls.weights, cls.Q = cls.coll.genCoeffs()
            cls.nodeType, cls.quadType = nodeType, quadType
        if updateQDeltaI:
            cls.QDeltaI, cls.dtauI = genQDeltaCoeffs(
                implSweep, dTau=True, nSweeps=nSweeps,
                Q=cls.Q, nodes=cls.nodes, nNodes=nNodes,
                nodeType=nodeType, quadType=quadType)
            cls.implSweep = implSweep
        if updateQDeltaE:
            cls.QDeltaE, cls.dtauE = genQDeltaCoeffs(
                explSweep, dTau=True, nSweeps=nSweeps,
                Q=cls.Q, nodes=cls.nodes, nNodes=nNodes,
                nodeType=nodeType, quadType=quadType)
            cls.explSweep = explSweep
        if updateQDelta0:
            cls.QDelta0 = genQDeltaCoeffs(
                initSweepQDelta,
                Q=cls.Q, nodes=cls.nodes, nNodes=nNodes,
                nodeType=nodeType, quadType=quadType)

        # Eventually update additional parameters
        if forceProl is not None: cls.forceProl = forceProl
        if initSweep is not None: cls.initSweep = initSweep
        if not keepNSweeps:
            cls.nSweeps = nSweeps

        diagonal = np.all([np.diag(np.diag(qD)) == qD for qD in cls.QDeltaI])
        diagonal *= np.all([np.diag(np.diag(qD)) == 0 for qD in cls.QDeltaE])
        if cls.initSweep == "QDELTA":
            diagonal *=  np.all(np.diag(np.diag(cls.QDelta0)) == cls.QDelta0)
        cls.diagonal = diagonal

    # -------------------------------------------------------------------------
    # Class properties
    # -------------------------------------------------------------------------
    @classmethod
    def getMaxOrder(cls):
        # TODO : adapt to non-LEGENDRE node distributions
        M = len(cls.nodes)
        return 2*M if cls.quadType == 'GAUSS' else \
            2*M-1 if cls.quadType.startswith('RADAU') else \
            2*M-2  # LOBATTO

    @classmethod
    def getInfos(cls):
        return sdcInfos(
            len(cls.nodes), cls.quadType, cls.nodeType, cls.nSweeps,
            cls.implSweep, cls.explSweep, cls.initSweep, cls.forceProl)

    @classmethod
    def getM(cls):
        return len(cls.nodes)

    @classmethod
    def rightIsNode(cls):
        return np.isclose(cls.nodes[-1], 1.0)

    @classmethod
    def leftIsNode(cls):
        return np.isclose(cls.nodes[0], 0.0)

    @classmethod
    def doProlongation(cls):
        return not cls.rightIsNode or cls.forceProl


if __name__ == '__main__':
    sdc = IMEXSDCCore()
