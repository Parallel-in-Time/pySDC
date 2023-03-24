#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 18:16:35 2023

@author: cpf5546
"""
import numpy as np
from time import process_time
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# Some performance tests
# -----------------------------------------------------------------------------
from pySDC.playgrounds.datatypes.base import DataTypeF
from pySDC.playgrounds.datatypes.implementations import Mesh
from pySDC.implementations.datatype_classes.mesh import imex_mesh, mesh

# -> RHS evaluation functions
def evalF_new(u, t):
    fEval = DataTypeF(u)
    fEval.values = -2*u + t**2
    return fEval

def evalF_ref(u, t):
    fEval = mesh(((u.size,), None, np.dtype('float64')))
    fEval[:] = -2*u + t**2
    return fEval

def evalF_new_IMEX(u, t):
    fEval = DataTypeF(u, parts=('impl', 'expl'))
    fEval.impl.values = -2*u + t**2
    fEval.expl.values = 1*u - t
    return fEval

def evalF_ref_IMEX(u, t):
    fEval = imex_mesh(((u.size,), None, np.dtype('float64')))
    fEval.expl[:] = -2*u + t**2
    fEval.expl[:] = 1*u - t
    return fEval


vecN = np.array([100, 500, 1000, 5000, 10000, 50000, 100000, 1000000, 2000000])
nRuns = 10*max(vecN)//vecN
res = np.zeros((8, len(vecN)))
for i, (n, nRun) in enumerate(zip(vecN, nRuns)):

    uNew = Mesh(shape=(n,), dtype=np.dtype('float64'))
    uNew.values = 1.0

    uRef = mesh(((n,), None, np.dtype('float64')))
    uRef[:] = 1.0

    for _ in range(nRun):
        tBeg = process_time()
        fNew = evalF_new(uNew, 0.1)
        tMid = process_time()
        uRes = uNew + 0.1*fNew
        tEnd = process_time()
        res[0, i] += tEnd-tBeg
        res[1, i] += tEnd-tMid

        tBeg = process_time()
        fNew_IMEX = evalF_new_IMEX(uNew, 0.1)
        tMid = process_time()
        uRes = uNew + 0.1*fNew
        tEnd = process_time()
        res[2, i] += tEnd-tBeg
        res[3, i] += tEnd-tMid

        tBeg = process_time()
        fRef = evalF_ref(uRef, 0.1)
        tMid = process_time()
        uRes = uRef + 0.1*fRef
        tEnd = process_time()
        res[4, i] += tEnd-tBeg
        res[5, i] += tEnd-tMid

        tBeg = process_time()
        fRef_IMEX = evalF_ref_IMEX(uRef, 0.1)
        tMid = process_time()
        uRes = uRef + 0.1*(fRef_IMEX.impl + fRef_IMEX.expl)
        tEnd = process_time()
        res[6, i] += tEnd-tBeg
        res[7, i] += tEnd-tMid

res /= nRuns

p = plt.loglog(vecN, res[0], label='u + dt*f(u,t)')
plt.loglog(vecN, res[4], '--', c=p[0].get_color())

p = plt.loglog(vecN, res[1], label='u + dt*f')
plt.loglog(vecN, res[5], '--', c=p[0].get_color())

p = plt.loglog(vecN, res[2], label='u + dt*f(u,t) (IMEX)')
plt.loglog(vecN, res[6], '--', c=p[0].get_color())

p = plt.loglog(vecN, res[3], label='u + dt*f (IMEX)')
plt.loglog(vecN, res[7], '--', c=p[0].get_color())

plt.legend()
plt.grid()
plt.xlabel('Vector size')
plt.ylabel('Computation time')
plt.tight_layout()
plt.show()
