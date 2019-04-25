import numpy as np
import time
from mpi4py import MPI
from mpi4py_fft import PFFT
from pySDC.playgrounds.mpifft.FFT_datatype import fft_datatype
from pySDC.implementations.datatype_classes.mesh import mesh


class mytype(np.ndarray):

    # def __new__(cls, global_shape, dtype=np.float, buffer=None):
    #     obj = np.ndarray.__new__(cls, global_shape, dtype=dtype, buffer=buffer)
    #     return obj

    def __new__(cls, init, val=0.0):
        if isinstance(init, mytype):
            obj = np.ndarray.__new__(cls, init.shape, dtype=init.dtype, buffer=None)
            obj[:] = init[:]
        elif isinstance(init, tuple):
            obj = np.ndarray.__new__(cls, init, dtype=np.float, buffer=None)
            obj[:] = val
        else:
            raise NotImplementedError(type(init))
        return obj

    def __abs__(self):
        return float(np.amax(np.ndarray.__abs__(self)))


nvars = 400
nruns = 100000

res = 0
t0 = time.perf_counter()
m = mytype((nvars, nvars), val=2.0)
for i in range(nruns):
    o = mytype(m)
    m[:] = 4.0
    res = max(res, abs(m + o))
t1 = time.perf_counter()
print(res, t1-t0)

res = 0
t0 = time.perf_counter()
m = mesh(init=(nvars, nvars), val=2.0)
for i in range(nruns):
    o = mesh(init=m)
    m.values[:] = 4.0
    res = max(res, abs(m + o))
t1 = time.perf_counter()
print(res, t1-t0)

exit()
# n = mytype((nvars, nvars), val=2.0)
m[:] = -1
n[:] = 2.9
print(n)
o = mytype(m)
print(o is m)
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


