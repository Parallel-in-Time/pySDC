import numpy as np

from pySDC.implementations.collocation_classes.generic import Collocation
from pySDC.implementations.collocation_classes.equidistant import \
    Equidistant
from pySDC.implementations.collocation_classes.equidistant_inner import \
    EquidistantInner
from pySDC.implementations.collocation_classes.equidistant_right import \
    EquidistantNoLeft
from pySDC.implementations.collocation_classes.equidistant_spline_right import \
    EquidistantSpline_Right
from pySDC.implementations.collocation_classes.gauss_legendre import \
    CollGaussLegendre
from pySDC.implementations.collocation_classes.gauss_lobatto import \
    CollGaussLobatto
from pySDC.implementations.collocation_classes.gauss_radau_left import \
    CollGaussRadau_Left
from pySDC.implementations.collocation_classes.gauss_radau_right import \
    CollGaussRadau_Right

EQUIV = {('EQUID', 'LOBATTO', False): Equidistant,
         ('EQUID', 'GAUSS', False): EquidistantInner,
         ('EQUID', 'RADAU-RIGHT', False): EquidistantNoLeft,
         ('EQUID', 'RADAU-RIGHT', True): EquidistantSpline_Right,
         ('LEGENDRE', 'GAUSS', False): CollGaussLegendre,
         ('LEGENDRE', 'LOBATTO', False): CollGaussLobatto,
         ('LEGENDRE', 'RADAU-LEFT', False): CollGaussRadau_Left,
         ('LEGENDRE', 'RADAU-RIGHT', False): CollGaussRadau_Right,}

def testEquivalencies():

    M = 5
    tLeft, tRight = 0, 1
    norm = lambda diff: np.linalg.norm(diff, ord=np.inf)
    tol = 1e-14

    lAttrVect = ['nodes', 'weights', 'Qmat', 'Smat', 'delta_m']
    lAttrScalar = ['order', 'left_is_node', 'right_is_node']

    # Compare each original class with their equivalent generic implementation
    for params, CollClass in EQUIV.items():
        cOrig = CollClass(M, tLeft, tRight)
        cNew = Collocation(M, tLeft, tRight, *params)
        for attr in lAttrVect:
            if not norm(getattr(cOrig, attr)-getattr(cNew, attr)) < tol:
                print(params)
        for attr in lAttrScalar:
            if not getattr(cOrig, attr) == getattr(cNew, attr):
                print(params)
