#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 16:28:32 2022

@author: cpf5546
"""
# Python imports
import numpy as np
import pandas as pd

# Local imports
from mplsettings import plt  # matplotlib with adapted font size settings
from qmatrix import genCollocation
from dahlquist import IMEXSDC
from optim import findLocalMinima

# ----------------------------------------------------------------------------
# Main parameters
# ----------------------------------------------------------------------------
# Collocation parameters
M = 3
distr = 'LEGENDRE'
quadType = 'LOBATTO'

# Optimization parameters
optimType = 'Speck'
# -- "Speck" : minimizes the spectral radius of I - QDelta^{-1}Q ... ImQdInvQ
# -- "QmQd" : minimizes the spectral radius of Q-QDelta
# -- "probDep" : minimizes the spectral radius of (I-QDelta)^{-1} (Q-QDelta)
# Value used for probDep optimization
lamDt = 1j

optimParams = {
    'nSamples': 200,        # Number of samples for initial optimization guess
    'nLocalOptim': 3,       # Number of local optimization for one sample
    'localOptimTol': 1e-9,  # Local optimization tolerance for one sample
    'randomSeed': None,     # Random seed to generate all samples
    'alphaFunc': 1.0,       # Exponant added to the optimization function
    'threshold': 1e-6}      # Threshold used to keep optimization candidates

# ----------------------------------------------------------------------------
# Script run
# ----------------------------------------------------------------------------
Q = genCollocation(M, distr, quadType)[2]

if quadType in ['RADAU-LEFT', 'LOBATTO']:
    dim = M-1
    Q = Q[1:,1:]
else:
    dim = M

if optimType == 'Speck':

    # Maximum inverse coefficient investigated
    maxCoeff = 13.

    def spectralRadius(x):
        QDeltaInv = np.diag(x)
        R = np.eye(dim) - QDeltaInv @ Q
        return np.max(np.abs(np.linalg.eigvals(R)))

elif optimType == 'QmQd':

    # Maximum coefficient investigated
    maxCoeff = 1.

    def spectralRadius(x):
        R = (Q - np.diag(x))
        return np.max(np.abs(np.linalg.eigvals(R)))

elif optimType == 'probDep':

    # Maximum coefficient investigated
    maxCoeff = 1.

    def spectralRadius(x):
        x = np.asarray(x)
        R = np.diag((1-lamDt*x)**(-1)) @ (Q - np.diag(x))
        return np.max(np.abs(np.linalg.eigvals(R)))
else:
    raise NotImplementedError(f'optimType={optimType}')

# Use Monte-Carlo Local Minima finder
res, xStarts = findLocalMinima(
    spectralRadius, dim, bounds=(0, maxCoeff), **optimParams)

# Manually compute Rho, and plot for 2D optimization
nPlotPts = int(50000**(1/dim))
limits = [maxCoeff]*dim
grids = np.meshgrid(*[
    np.linspace(0, l, num=nPlotPts) for l in limits])
flatGrids = np.array([g.ravel() for g in grids])

# Computing spectral radius for many coefficients
print('Computing spectral radius values')
rho = []
for coeffs in flatGrids.T:
    rho.append(spectralRadius(coeffs))
rho = np.reshape(rho, grids[0].shape)
rho[rho>1] = 1

if dim == 2:
    plt.figure()
    plt.contourf(*grids, rho,
                 levels=np.linspace(0, 1, 101), cmap='OrRd')
    plt.colorbar(ticks=np.linspace(0, 1, 11))
    for x0 in xStarts:
        plt.plot(*x0, '+', c='gray', ms=6, alpha=0.75)
    for xMin in res:
        plt.plot(*xMin, 'o', ms=12)
        pos = [1.1*x for x in xMin]
        plt.text(*pos, r'$\rho_{opt}='f'{res[xMin]:1.2e}$',
                 fontsize=16)

    if optimType == 'QmQd' and False:
        # Only if you want to check A-stability ...
        print('Computing imaginary stability')
        iStab = []
        IMEXSDC.setParameters(
            M=M, quadType=quadType, nodeDistr=distr, implSweep='BEpar',
            initSweep='COPY', forceProl=True)
        for coeffs in flatGrids.T:
            if quadType in ['RADAU-LEFT', 'LOBATTO']:
                IMEXSDC.QDeltaI[1:,1:] = np.diag(coeffs)
            else:
                IMEXSDC.QDeltaI[:] = np.diag(coeffs)
            iStab.append(IMEXSDC.imagStability())
        iStab = np.reshape(iStab, grids[0].shape)

        plt.figure()
        plt.contourf(*grids, iStab,
                     levels=np.linspace(0, 20, 101), cmap='OrRd')
        plt.colorbar(ticks=np.linspace(0, 20, 11))


if optimType == 'Speck':
    # For Speck optim type, optimization produces the coefficient of the
    # inverse of the QDelta matrix
    for xOpt in list(res.keys()):
        rho = res.pop(xOpt)
        xOpt = tuple(1/x for x in xOpt)
        res[xOpt] = rho

print('Optimum diagonal coefficients found :')
print(f' -- optimType={optimType}')
for xOpt in res:
    print(f' -- xOpt={xOpt} (rho={res[xOpt]:1.2e})')

# Store results in Markdown dataframe
if optimType != 'probDep':
    try:
        df = pd.read_table(
            'optimDiagCoeffs.md', sep="|", header=0, index_col=0,
            skipinitialspace=True).dropna(axis=1, how='all').iloc[1:]
        df.reset_index(inplace=True, drop=True)
        df.columns = [label.strip() for label in df.columns]
        df = df.applymap(lambda x: x.strip())
        df['rho'] = df.rho.astype(float)
        df['M'] = df.M.astype(int)
        df['coeffs'] = df.coeffs.apply(
            lambda x: tuple(float(n) for n in x[1:-1].split(', ')))
    except Exception:
        df = pd.DataFrame(
            columns=['M', 'quadType', 'distr', 'optim', 'coeffs', 'rho'])

    def formatCoeffs(c):
        out = tuple(round(v, 6) for v in c)
        if quadType in ['RADAU-LEFT', 'LOBATTO']:
            out = (0.,) + out
        return out

    def addCoefficients(line, df):
        cond = (df == line.values)
        l = df[cond].iloc[:, :-1].dropna()
        if l.shape[0] == 1:
            # Coefficients already stored
            idx = l.index[0]
            if line.rho[0] < df.loc[idx, 'rho']:
                df.loc[idx, 'rho'] = line.rho[0]
        else:
            # New computed coefficients
            df = pd.concat((df, line), ignore_index=True)
        return df

    for c in res:
        line = pd.DataFrame(
            [[M, quadType, distr, optimType, formatCoeffs(c), res[c]]],
            columns=df.columns)
        df = addCoefficients(line, df)

    df.sort_values(by=['M', 'quadType', 'optim', 'coeffs'], inplace=True)
    df.to_markdown(buf='optimDiagCoeffs.md', index=False)
