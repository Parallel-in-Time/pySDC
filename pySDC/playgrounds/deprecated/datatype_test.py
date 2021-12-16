import time
import numpy as np
from dedalus import public as de

from pySDC.implementations.datatype_classes.mesh import mesh
from pySDC.implementations.datatype_classes.mesh import mesh
from pySDC.playgrounds.Dedalus.dedalus_field import dedalus_field
from pySDC.playgrounds.Dedalus.dedalus_field_fast import dedalus_field_fast




def inner_loop(iloops, dtype, maxb, a, b):
    for i in range(iloops):
        b += 0.1 / (i + 1) * a
        print(type(b))
        a = dtype(b)
    b = 1.0 / iloops * b
    maxb = max(maxb, abs(b))
    return maxb


def mesh_test(oloops, iloops, N):

    maxtime = 0.0
    mintime = 1E99
    meantime = 0.0
    maxb = 0.0
    for i in range(oloops):

        t0 = time.perf_counter()
        a = mesh(init=N, val=1.0 * i)
        b = mesh(a)
        maxb = inner_loop(iloops, mesh, maxb, a, b)
        t1 = time.perf_counter()

        maxtime = max(maxtime, t1 - t0)
        mintime = min(mintime, t1 - t0)
        meantime += t1 - t0
    meantime /= oloops
    print(maxb)
    print(maxtime, mintime, meantime)


def parallelmesh_test(oloops, iloops, N):

    maxtime = 0.0
    mintime = 1E99
    meantime = 0.0
    maxb = 0.0
    for i in range(oloops):

        t0 = time.perf_counter()
        a = mesh(init=(N, None, np.zeros(1).dtype), val=1.0 * i)
        b = mesh(a)
        maxb = inner_loop(iloops, mesh, maxb, a, b)
        t1 = time.perf_counter()

        maxtime = max(maxtime, t1 - t0)
        mintime = min(mintime, t1 - t0)
        meantime += t1 - t0
    meantime /= oloops
    print(maxb)
    print(maxtime, mintime, meantime)


def field_test(oloops, iloops, N):

    xbasis = de.Fourier('x', N[0], interval=(-1, 1), dealias=1)
    ybasis = de.Fourier('y', N[1], interval=(-1, 1), dealias=1)
    domain = de.Domain([xbasis, ybasis], grid_dtype=np.float64, comm=None)

    maxtime = 0.0
    mintime = 1E99
    meantime = 0.0
    maxb = 0.0
    for i in range(oloops):

        t0 = time.perf_counter()
        a = dedalus_field(init=domain)
        a.values[0]['g'][:] = 1.0 * i
        b = dedalus_field(a)
        maxb = inner_loop(iloops, dedalus_field, maxb, a, b)
        t1 = time.perf_counter()

        maxtime = max(maxtime, t1 - t0)
        mintime = min(mintime, t1 - t0)
        meantime += t1 - t0
    meantime /= oloops
    print(maxb)
    print(maxtime, mintime, meantime)


def fast_field_test(oloops, iloops, N):

    xbasis = de.Fourier('x', N[0], interval=(-1, 1), dealias=1)
    ybasis = de.Fourier('y', N[1], interval=(-1, 1), dealias=1)
    domain = de.Domain([xbasis, ybasis], grid_dtype=np.float64, comm=None)

    maxtime = 0.0
    mintime = 1E99
    meantime = 0.0
    maxb = 0.0
    for i in range(oloops):

        t0 = time.perf_counter()
        a = dedalus_field_fast(init=domain)
        a['g'][:] = 1.0 * i
        b = dedalus_field_fast(a)
        maxb = inner_loop(iloops, dedalus_field_fast, maxb, a, b)
        t1 = time.perf_counter()

        maxtime = max(maxtime, t1 - t0)
        mintime = min(mintime, t1 - t0)
        meantime += t1 - t0
    meantime /= oloops
    print(maxb)
    print(maxtime, mintime, meantime)

if __name__ == '__main__':

    oloops = 1
    iloops = 1

    N = (64, 64)

    # mesh_test(oloops, iloops, N)
    # parallelmesh_test(oloops, iloops, N)
    # field_test(oloops, iloops, N)
    fast_field_test(oloops, iloops, N)