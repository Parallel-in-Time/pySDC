import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl
import matplotlib.pyplot as plt

import pySDC.helpers.transfer_helper as th

np.random.seed(32)

nX = 2 ** 3 - 1
nXc = 2 ** 2 - 1
dx = 1/(nX + 1)
dxc = 1/(nXc + 1)

stencil = [1, -2, 1]
A = sp.diags(stencil, [-1, 0, 1], shape=(nX, nX), format='csc')
A *= - 1.0 / (dx ** 2)
b = np.zeros(nX)
# %%
PGS = spl.inv(sp.csc_matrix(np.tril(A.todense())))
PJ = sp.dia_matrix(np.diag(1/np.diag(A.todense())))

RGS = sp.eye(nX) - PGS @ A
RJ = sp.eye(nX) - PJ @ A
sRJ = sp.eye(nX) - 0.5 * PJ @ A

Ac = sp.diags(stencil, [-1, 0, 1], shape=(nX//2, nX//2), format='csc')
Ac *= - 1.0 / (dxc ** 2)
Acinv = spl.inv(Ac)
PJc = sp.dia_matrix(np.diag(1/np.diag(Ac.todense())))

fine_grid = np.linspace(dx, 1, nX, endpoint=False)
coarse_grid = np.linspace(dxc, 1, nXc, endpoint=False)
Pr = th.interpolation_matrix_1d(fine_grid, coarse_grid, k=2, periodic=False, equidist_nested=True)
Prinj = th.interpolation_matrix_1d(fine_grid, coarse_grid, k=0, periodic=False, equidist_nested=True)
Re = 0.5 * Pr.T
# Re = Prinj.T

TG = (sp.eye(nX) - Pr @ Acinv @ Re @ A) @ sRJ.power(3)
TJ = (sp.eye(nX) - Pr @ PJc @ Re @ A) @ sRJ.power(3)

# %%

nIter = 100
dK = 10

def coarse(x, dK=dK):
    xK = x
    for k in range(dK):
        xK = RJ @ xK #+ RJ @ b
    return xK


def fine(x, dK=dK):
    xK = x
    for k in range(dK):
        xK = TG @ xK #+ PGS @ b
    return xK

def initBlocks(x):
    x0 = np.sin(np.linspace(dx, 1*2*np.pi, nX))
    # x0 = np.random.rand(nX)
    for l in range(nB+1):
        x[l, :] = x0

nB = nIter//dK
nK = nB+1


uSeq = np.zeros((nB+1, nX))
uNum = np.zeros((nK+1, nB+1, nX))

initBlocks(uNum[0])

uSeq[0] = uNum[0, 0]
uNum[:, 0, :] = uNum[0, 0, :]

for k in range(nK):
    for l in range(nB):

        uF = fine(uNum[k, l])
        uGk = coarse(uNum[k, l], dK=1)
        uGkp1 = coarse(uNum[k+1, l], dK=1)

        uNum[k+1, l+1] = uF + 1*(uGkp1 - uGk)


for l in range(nB):
    uSeq[l+1] = fine(uSeq[l])

# %%

err = np.linalg.norm(uNum, axis=-1)[:, -1]
errSeq = np.linalg.norm(uSeq, axis=-1)

plt.figure()
plt.semilogy(err, label='Parareal')
plt.semilogy(errSeq, label='Sequential')
plt.legend()
plt.show()
