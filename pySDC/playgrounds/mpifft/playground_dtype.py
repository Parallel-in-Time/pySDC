import numpy as np
import time
import copy
from mpi4py import MPI
from mpi4py_fft import PFFT
from pySDC.playgrounds.mpifft.FFT_datatype import fft_datatype
from pySDC.implementations.datatype_classes.mesh import mesh


class mytype(np.ndarray):

    def __new__(cls, init, val=0.0):
        if isinstance(init, mytype):
            obj = np.ndarray.__new__(cls, init.shape, dtype=np.float, buffer=None)
            obj[:] = init[:]
            # obj = init.copy()
        elif isinstance(init, tuple):
            obj = np.ndarray.__new__(cls, init, dtype=np.float, buffer=None)
            obj.fill(val)
        else:
            raise NotImplementedError(type(init))
        return obj

    def __abs__(self):
        return float(np.amax(np.ndarray.__abs__(self)))


nvars = 32
nruns = 100 * 16

res = 0
t0 = time.perf_counter()
m = mytype((nvars, nvars), val=2.0)
for i in range(nruns):
    o = mytype(m)
    o[:] = 4.0
    n = mytype(m)
    for j in range(i):
        n += o
    res = max(res, abs(n))
t1 = time.perf_counter()
print(res, t1-t0)

# res = 0
# t0 = time.perf_counter()
# m = mytype((nvars, nvars), val=2.0)
# for i in range(nruns):
#     o = mytype(m)
#     o.values[:] = 4.0
#     res = max(res, abs(m + o))
# t1 = time.perf_counter()
# print(res, t1-t0)

res = 0
t0 = time.perf_counter()
m = mesh(init=(nvars, nvars), val=2.0)
for i in range(nruns):
    o = mesh(init=m)
    o.values[:] = 4.0
    n = mesh(init=m)
    for j in range(i):
        n += o
    res = max(res, abs(n))
t1 = time.perf_counter()
print(res, t1-t0)

m = mytype((nvars, nvars), val=2.0)
n = mytype((nvars, nvars), val=2.0)
m[:] = -1
n[:] = 2.9
# print(n)
o = mytype(m)
m[0, 0] = 2
assert o[0, 0] == -1
assert o is not m
exit()

print(type(m))
print(type(m+n))
print(abs(m))
print(type(abs(m)))

comm = MPI.COMM_WORLD
subcomm = comm.Split()
# print(subcomm)
nvars = 8
ndim = 2
axes = tuple(range(ndim))
N = np.array([nvars] * ndim, dtype=int)
# print(N, axes)
fft = PFFT(subcomm, N, axes=axes, dtype=np.float, slab=True)

init = (fft, False)
m = fft_datatype(init)
m[:] = comm.Get_rank()

print(type(m))
print(m.subcomm)
print(abs(m), type(abs(m)))


