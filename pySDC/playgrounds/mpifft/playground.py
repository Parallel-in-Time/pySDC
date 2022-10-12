import numpy as np
from mpi4py import MPI
from mpi4py_fft import PFFT, newDistArray, DistArray
import time
import copy as cp


def get_local_mesh(FFT, L):
    """Returns local mesh."""
    X = np.ogrid[FFT.local_slice(False)]
    N = FFT.global_shape()
    for i in range(len(N)):
        X[i] = X[i] * L[i] / N[i]
    X = [np.broadcast_to(x, FFT.shape(False)) for x in X]
    return X


def get_local_wavenumbermesh(FFT, L):
    """Returns local wavenumber mesh."""
    s = FFT.local_slice()
    N = FFT.global_shape()
    # Set wavenumbers in grid
    k = [np.fft.fftfreq(n, 1.0 / n).astype(int) for n in N[:-1]]
    k.append(np.fft.rfftfreq(N[-1], 1.0 / N[-1]).astype(int))
    K = [ki[si] for ki, si in zip(k, s)]
    Ks = np.meshgrid(*K, indexing='ij', sparse=True)
    Lp = 2 * np.pi / L
    for i in range(ndim):
        Ks[i] = (Ks[i] * Lp[i]).astype(float)
    return [np.broadcast_to(k, FFT.shape(True)) for k in Ks]


comm = MPI.COMM_WORLD
subcomm = comm.Split()
print(subcomm)
nvars = 8
ndim = 2
axes = tuple(range(ndim))
N = np.array([nvars] * ndim, dtype=int)
print(N, axes)
fft = PFFT(subcomm, N, axes=axes, dtype=np.float, slab=True)
# L = np.array([2*np.pi] * ndim, dtype=float)
L = np.array([1] * ndim, dtype=float)

print(fft.subcomm)

X = get_local_mesh(fft, L)
K = get_local_wavenumbermesh(fft, L)
K = np.array(K).astype(float)
K2 = np.sum(K * K, 0, dtype=float)

u = newDistArray(fft, False)
print(type(u))
print(u.subcomm)
uex = newDistArray(fft, False)

u[:] = np.sin(2 * np.pi * X[0]) * np.sin(2 * np.pi * X[1])
print(u.shape, X[0].shape)
# exit()
uex[:] = -2.0 * (2.0 * np.pi) ** 2 * np.sin(2 * np.pi * X[0]) * np.sin(2 * np.pi * X[1])
u_hat = fft.forward(u)

lap_u_hat = -K2 * u_hat

lap_u = np.zeros_like(u)
lap_u = fft.backward(lap_u_hat, lap_u)
local_error = np.amax(abs(lap_u - uex))
err = MPI.COMM_WORLD.allreduce(local_error, MPI.MAX)
print('Laplace error:', err)

ratio = 2
Nc = np.array([nvars // ratio] * ndim, dtype=int)
fftc = PFFT(MPI.COMM_WORLD, Nc, axes=axes, dtype=np.float, slab=True)
print(Nc, fftc.global_shape())
Xc = get_local_mesh(fftc, L)

uex = newDistArray(fft, False)
uexc = newDistArray(fftc, False)
r2 = (X[0] - 0.5) ** 2 + (X[1] - 0.5) ** 2
uex[:] = 0.5 * (1.0 + np.tanh((0.25 - np.sqrt(r2)) / (np.sqrt(2) * 0.04)))
r2c = (Xc[0] - 0.5) ** 2 + (Xc[1] - 0.5) ** 2
uexc[:] = 0.5 * (1.0 + np.tanh((0.25 - np.sqrt(r2c)) / (np.sqrt(2) * 0.04)))

uc = uex[::ratio, ::ratio]
local_error = np.amax(abs(uc - uexc))
err = MPI.COMM_WORLD.allreduce(local_error, MPI.MAX)
print('Restriction error real:', err)

uexcs = fftc.forward(uexc)
uc = uex[::ratio, ::ratio]
ucs = fftc.forward(uc)
local_error = np.amax(abs(ucs - uexcs))
err = MPI.COMM_WORLD.allreduce(local_error, MPI.MAX)
print('Restriction error spectral:', err)

uexc_hat = fftc.forward(uexc)
fft_pad = PFFT(MPI.COMM_WORLD, Nc, padding=[ratio] * ndim, axes=axes, dtype=np.float, slab=True)
uf = newDistArray(fft_pad, False)
uf = fft_pad.backward(uexc_hat, uf)
local_error = np.amax(abs(uf - uex))
err = MPI.COMM_WORLD.allreduce(local_error, MPI.MAX)
print('Interpolation error real:', err)


uexs = fft.forward(uex)
fft_pad = PFFT(MPI.COMM_WORLD, Nc, padding=[ratio] * ndim, axes=axes, dtype=np.float, slab=True)
# uf = fft_pad.backward(uexc_hat)
# ufs = fft.forward(uf)
ufs = np.pad(uexc_hat, [(0, Nc[0]), (0, Nc[1] // 2)], mode='constant')
# ufs[:][0] *= 2
print(uexc_hat[1])
print(uexs[1])
print(uexc_hat.shape, ufs.shape, uexs.shape)

local_error = np.amax(abs(ufs / 4 - uexs))
err = MPI.COMM_WORLD.allreduce(local_error, MPI.MAX)
print('Interpolation error spectral:', err)
exit()

u = newDistArray(fft, False)
ucopy = u.copy()
print(type(u), type(ucopy), u is ucopy)
u[0] = 1
print(ucopy[0, 0])
print(np.all(u == ucopy))

nruns = 30000
s = 0
t0 = time.perf_counter()
for i in range(nruns):
    u = newDistArray(fft, False, val=i)
    v = u.copy()
    u[0][0] = 0
    # print(u[0][0], v[0][0])
    s += u[0][0] - v[0][0]
t1 = time.perf_counter()
print(s + nruns * (nruns - 1) / 2, t1 - t0)

s = 0
t0 = time.perf_counter()
for i in range(nruns):
    u = np.full((nvars, nvars), fill_value=i, dtype=np.float64)
    # u = np.empty((nvars, nvars), dtype=np.float64)
    # u[:] = i
    v = u.copy()
    # v = cp.deepcopy(u)
    u[0][0] = 0
    # print(u[0][0], v[0][0])
    s += u[0][0] - v[0][0]
t1 = time.perf_counter()
print(s + nruns * (nruns - 1) / 2, t1 - t0)
