#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 14:52:18 2023

@author: telu
"""
import pytest
import numpy as np

from pySDC.core.Lagrange import LagrangeApproximation

# Pre-compute reference integration matrix
nNodes = 5
approx = LagrangeApproximation(np.linspace(0, 1, nNodes))
nIntegPoints = 13
tEndVals = np.linspace(0, 1, nIntegPoints)
integMatRef = approx.getIntegrationMatrix([(0, t) for t in tEndVals])


@pytest.mark.base
@pytest.mark.parametrize("numQuad", ["LEGENDRE_NUMPY", "LEGENDRE_SCIPY"])
def test_numericalQuadrature(numQuad):
    integMat = approx.getIntegrationMatrix([(0, t) for t in tEndVals], numQuad=numQuad)
    assert np.allclose(integMat, integMatRef)
