import numpy as np
import matplotlib.pyplot as plt

from pySDC.implementations.collocation_classes.gauss_lobatto import CollGaussLobatto
from pySDC.implementations.collocation_classes.gauss_radau_left import CollGaussRadau_Left
from pySDC.implementations.collocation_classes.gauss_radau_right import CollGaussRadau_Right
from pySDC.implementations.collocation_classes.equidistant import Equidistant

collDict = {
    'LOBATTO': CollGaussLobatto,
    'RADAU_LEFT': CollGaussRadau_Left,
    'RADAU_RIGHT': CollGaussRadau_Right,
    'EQUID': Equidistant,
}

# Problem parameters
tBeg = 0
tEnd = 1
lam = -1.0 + 1j  # \lambda coefficient from the space operator
numType = np.array(lam).dtype
u0 = 1  # initial solution

# Discretization parameters
L = 8  # number of time steps
M = 5  # number of nodes
sweepType = 'BE'
nodesType = 'EQUID'

# Time interval decomposition
times = np.linspace(tBeg, tEnd, num=L)
dt = times[1] - times[0]

# Generate nodes, deltas and Q using pySDC
coll = collDict[nodesType](M, 0, 1)
nodes = coll._getNodes
deltas = coll._gen_deltas
Q = coll._gen_Qmatrix[1:, 1:]
Q = Q * lam * dt

# Generate Q_\Delta
QDelta = np.zeros((M, M))
if sweepType in ['BE', 'FE']:
    offset = 1 if sweepType == 'FE' else 0
    for i in range(offset, M):
        QDelta[i:, : M - i] += np.diag(deltas[: M - i])
else:
    raise ValueError(f'sweepType={sweepType}')
QDelta = QDelta * lam * dt

print(QDelta.real)
exit()

# Generate other matrices
H = np.zeros((M, M), dtype=numType)
H[:, -1] = 1
I = np.identity(M)

# Generate operators
ImQDelta = I - QDelta
ImQ = I - Q


def fineSweep(u, uStar, f):
    u += np.linalg.solve(ImQDelta, H.dot(uStar) + f - ImQ.dot(u))


def coarseSweep(u, uStar, f):
    u += np.linalg.solve(ImQDelta, H.dot(uStar) + f)


# Variables
f = np.zeros((L, M), dtype=numType)
f[0] = u0
u = np.zeros((L, M), dtype=numType)
t = np.array([t + dt * nodes for t in times])

# Exact solution
uExact = np.exp(t * lam)


nSweep = 3  # also number of iteration for Parareal-SDC and PFASST

# SDC solution
uSDC = u.copy()
# uSDC[:] = u0
uStar = u[0] * 0
for l in range(L):
    for n in range(nSweep):
        fineSweep(uSDC[l], uStar, f[l])
    uStar = uSDC[l]

# Parareal-SDC solution <=> pipelined SDC <=> RIDC ?
uPar = u.copy()
# uPar[:] = u0
for n in range(nSweep):
    uStar = u[0] * 0
    for l in range(L):
        fineSweep(uPar[l], uStar, f[l])
        uStar = uPar[l]

# PFASST solution
uPFA = u.copy()
if True:
    # -- initialization with coarse sweep
    # (equivalent to fine sweep, since uPFA[l] = 0)
    uStar = u[0] * 0
    for l in range(L):
        coarseSweep(uPFA[l], uStar, f[l])
        uStar = uPFA[l]
# -- PFASST iteration
for n in range(nSweep):
    uStar = u[0] * 0
    for l in range(L):
        uStarPrev = uPFA[l].copy()
        fineSweep(uPFA[l], uStar, f[l])
        uStar = uStarPrev

# Plot solution and error
if True:
    plt.figure('Solution (amplitude)')
    for sol, lbl, sym in [
        [uExact, 'Exact', 'o-'],
        [uSDC, 'SDC', 's-'],
        [uPar, 'Parareal', '>-'],
        [uPFA, 'PFASST', 'p-'],
    ]:
        plt.plot(t.ravel(), sol.ravel().real, sym, label=lbl)
    plt.legend()
    plt.grid(True)
    plt.xlabel('Time')

    plt.figure('Solution (phase)')
    for sol, lbl, sym in [
        [uExact, 'Exact', 'o-'],
        [uSDC, 'SDC', 's-'],
        [uPar, 'Parareal', '>-'],
        [uPFA, 'PFASST', 'p-'],
    ]:
        plt.plot(t.ravel(), sol.ravel().imag, sym, label=lbl)
    plt.legend()
    plt.grid(True)
    plt.xlabel('Time')

eSDC = np.abs(uExact.ravel() - uSDC.ravel())
ePar = np.abs(uExact.ravel() - uPar.ravel())
ePFA = np.abs(uExact.ravel() - uPFA.ravel())

plt.figure('Error')
plt.semilogy(t.ravel(), eSDC, 's-', label=f'SDC {nSweep} sweep')
plt.semilogy(t.ravel(), ePar, '>-', label=f'Parareal {nSweep} iter')
plt.semilogy(t.ravel(), ePFA, 'p-', label=f'PFASST {nSweep} iter')
plt.legend()
plt.grid(True)
plt.xlabel('Time')

plt.show()
