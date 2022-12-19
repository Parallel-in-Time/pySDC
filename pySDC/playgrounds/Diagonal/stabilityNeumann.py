#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 12:49:22 2022

Plot the stability contour of any flavor of SDC
SDC parameters can be set directly in the plotStabContour function ...

@author: cpf5546
"""
import numpy as np
import matplotlib.pyplot as plt

from dahlquist import IMEXSDC

u0 = 1.0
zoom = 5
lamReals = -5*zoom, 1*zoom, 301
lamImags = -4*zoom, 4*zoom, 400

xLam = np.linspace(*lamReals)[:, None]
yLam = np.linspace(*lamImags)[None, :]

lams = xLam + 1j*yLam

plt.figure()

def plotStabContour(nSweep):
    IMEXSDC.setParameters(
        M=2, quadType='GAUSS', nodeDistr='LEGENDRE',
        implSweep='WEIRD', explSweep='FE', initSweep='COPY',
        forceProl=True)
    IMEXSDC.nSweep = nSweep

    solver = IMEXSDC(u0, lams.ravel(), 0)
    solver.step(1.)

    uNum = solver.u.reshape(lams.shape)

    stab = np.abs(uNum)
    coords = np.meshgrid(xLam.ravel(), yLam.ravel(), indexing='ij')

    CS = plt.contour(*coords, stab, levels=[1.0], colors='k')
    plt.clabel(CS, inline=True, fmt=f'K={nSweep}')
    plt.grid(True)
    plt.gca().set_aspect('equal', 'box')
    plt.xlabel(r'$Re(\lambda)$')
    plt.ylabel(r'$Im(\lambda)$')
    plt.gcf().set_size_inches(4, 5)
    plt.title(IMEXSDC.implSweep)
    plt.tight_layout()

    return stab

for nSweep in [1, 2]:
    stab = plotStabContour(nSweep)
