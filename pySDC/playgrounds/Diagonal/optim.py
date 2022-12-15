#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 10:19:55 2022

@author: cpf5546
"""
import numpy as np
import scipy.optimize as sco
import skopt


def findLocalMinima(func, dim, bounds=(0, 15),
                     nSamples=200, nLocalOptim=3, localOptimTol=1e-8,
                     alphaFunc=1, randomSeed=None, threshold=1e-2):
    print('Monte-Carlo local minima finder')

    # Initialize random seed
    np.random.seed(randomSeed)

    # Compute starting points
    print(' -- generating random samples (Maximin Optimized Latin Hypercube)')
    space = skopt.space.Space([bounds]*dim)
    lhs = skopt.sampler.Lhs(criterion="maximin", iterations=10000)
    xStarts = lhs.generate(space.dimensions, nSamples)

    # Optimization functional
    modFunc = lambda x: func(x)**alphaFunc

    res = {}

    def findRes(x):
        x = [round(v, 6) for v in x]
        for x0 in res:
            if np.allclose(x, x0, rtol=1e-1):
                return x0
        return False

    # Look at randomly generated staring points
    print(' -- running local optimizations')
    for x0 in xStarts:

        # Run one or several local optimization
        for _ in range(nLocalOptim):
            opt = sco.minimize(modFunc, x0,
                               method='Nelder-Mead', tol=localOptimTol)
            x0 = opt.x
            if not opt.success:
                break

        funcEval = func(x0)

        # Eventually add local minimum to results
        xOrig = findRes(x0)
        if xOrig:
            if funcEval < res[xOrig]:
                res.pop(xOrig)
                res[tuple(x0)] = funcEval
                print('     -- found better local minimum')
        else:
            if funcEval < threshold:
                print('     -- found new local minimum')
                res[tuple(x0)] = funcEval

    return res, xStarts
