from mpi4py import MPI
import matplotlib
matplotlib.use("TkAgg")
import numpy as np
import time
from pmesh.pm import ParticleMesh

from pySDC.playgrounds.pmesh.PMESH_datatype import pmesh_datatype, rhs_imex_pmesh

def doublesine(i, v):
    r = [ii * (Li / ni) for ii, ni, Li in zip(i, v.Nmesh, v.BoxSize)]
    # xx, yy = np.meshgrid(r[0], r[1])
    return np.sin(2*np.pi*r[0]) * np.sin(2*np.pi*r[1])

def Lap_doublesine(i, v):
    r = [ii * (Li / ni) for ii, ni, Li in zip(i, v.Nmesh, v.BoxSize)]
    # xx, yy = np.meshgrid(r[0], r[1])
    return -2.0 * (2.0 * np.pi) ** 2 * np.sin(2*np.pi*r[0]) * np.sin(2*np.pi*r[1])

def Laplacian(k, v):
    k2 = sum(ki ** 2 for ki in k)
    # print([type(ki[0][0]) for ki in k])
    # k2[k2 == 0] = 1.0
    return -k2 * v

def circle(i, v):
    r = [ii * (Li / ni) - 0.5 * Li for ii, ni, Li in zip(i, v.Nmesh, v.BoxSize)]
    r2 = sum(ri ** 2 for ri in r)
    return np.tanh((0.25 - np.sqrt(r2)) / (np.sqrt(2) * 0.04))


nvars = 128
nruns = 1000

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

t0 = time.time()
pm = ParticleMesh(BoxSize=1.0, Nmesh=[nvars] * 2, dtype='f8', plan_method='measure', comm=comm)

u = pmesh_datatype(pm)
u.values.apply(circle, kind='index', out=Ellipsis)
v = pmesh_datatype(u)
v.values.value[0] = 1
print(np.amax(abs(u-v)))
exit()
t1 = time.time()

print(f'PMESH setup time: {t1 - t0:6.4f} sec.')

dt = 0.121233
res = 0.0
t0 = time.time()
for n in range(nruns):

    # set initial condition
    # u = pmesh_datatype(init=pm)
    # u.values.apply(circle, kind='index', out=Ellipsis)
    u = rhs_imex_pmesh(init=pm)
    u.impl.values.apply(circle, kind='index', out=Ellipsis)

    # save initial condition
    u_old = pmesh_datatype(init=u.impl)

    dt = 0.121233 / (n + 1)

    def linear_solve(k, v):
        global dt
        k2 = sum(ki ** 2 for ki in k)
        factor = 1 + dt * k2
        return 1.0 / factor * v

    # solve (I-dt*A)u = u_old
    sol = pmesh_datatype(init=pm)
    sol.values = u.impl.values.r2c().apply(linear_solve, out=Ellipsis).c2r(out=Ellipsis)

    # compute Laplacian
    # lap = sol.values.r2c().apply(Laplacian, out=Ellipsis).c2r(out=Ellipsis)
    lap = pmesh_datatype(init=pm)
    lap.values = sol.values.r2c().apply(Laplacian, out=Ellipsis).c2r(out=Ellipsis)

    # compute residual of (I-dt*A)u = u_old
    res = max(abs(sol - dt*lap - u_old), res)
t1 = time.time()

print(f'PMESH residual: {res:6.4e}')  # Should be approx. 5.9E-11
print(f'PMESH runtime: {t1 - t0:6.4f} sec.')  # Approx. 0.9 seconds on my machine

exit()

pmf = ParticleMesh(BoxSize=1.0, Nmesh=[nvars] * 2, comm=comm)
pmc = ParticleMesh(BoxSize=1.0, Nmesh=[nvars//2] * 2, comm=comm)

uexf = pmf.create(type='real')
uexf = uexf.apply(doublesine, kind='index')

uexc = pmc.create(type='real')
uexc = uexc.apply(doublesine, kind='index')

# uc = pmc.upsample(uexf, keep_mean=True)
uc = pmc.create(type='real')
uexf.resample(uc)
print(uc.preview().shape, np.amax(abs(uc-uexc)))

uf = pmf.create(type='real')
uexc.resample(uf)
print(uf.preview().shape, np.amax(abs(uf-uexf)))
print()

t1 = time.time()
print(f'Time: {t1-t0}')
