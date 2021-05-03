from mpi4py import MPI
import matplotlib
matplotlib.use("TkAgg")

import numpy as np
import pfft
import time
import matplotlib.pyplot as plt
from numpy.fft import rfft2, irfft2

from pmesh.pm import ParticleMesh, RealField, ComplexField

def doublesine(i, v):
    r = [ii * (Li / ni) for ii, ni, Li in zip(i, v.Nmesh, v.BoxSize)]
    # xx, yy = np.meshgrid(r[0], r[1])
    return np.sin(2*np.pi*r[0]) * np.sin(2*np.pi*r[1])

nvars = 128
nruns = 1

# t0 = time.time()
# pm = ParticleMesh(BoxSize=1.0, Nmesh=[nvars] * 2, dtype='f8', plan_method='measure', comm=None)
# u = pm.create(type='real')
# u = u.apply(doublesine, kind='index', out=Ellipsis)
#
# res = 0
# for i in range(nruns):
#     tmp = u.preview()
#     # print(type(u.value))
#     res = max(res, np.linalg.norm(tmp))
# print(res)
# t1 = time.time()
#
# print(f'PMESH setup time: {t1 - t0:6.4f} sec.')
#
# exit()

comm = MPI.COMM_WORLD

world_rank = comm.Get_rank()
world_size = comm.Get_size()
# split world communicator to create space-communicators
color = int(world_rank / 2)
space_comm = comm.Split(color=color)
space_size = space_comm.Get_size()
space_rank = space_comm.Get_rank()
color = int(world_rank % 2)
time_comm = comm.Split(color=color)
time_size = time_comm.Get_size()
time_rank = time_comm.Get_rank()

print(world_rank, time_rank, space_rank)

t0 = time.time()

if time_rank == 0:
    pm = ParticleMesh(BoxSize=1.0, Nmesh=[nvars] * 2, dtype='f8', plan_method='measure', comm=space_comm)
    u = pm.create(type='real')
    u = u.apply(doublesine, kind='index', out=Ellipsis)
    time_comm.send(u.value, dest=1, tag=11)
else:
    pm = ParticleMesh(BoxSize=1.0, Nmesh=[nvars] * 2, dtype='f8', plan_method='measure', comm=space_comm)
    tmp = time_comm.recv(source=0, tag=11)
    u = pm.create(type='real', value=tmp)


t1 = time.time()

print(f'PMESH setup time: {t1 - t0:6.4f} sec.')
exit()





