from mpi4py import MPI
import numpy as np
import time


def sleep(n):
    tmp = np.random.rand(n)


comm = MPI.COMM_WORLD
rank = comm.Get_rank()

t0 = time.perf_counter()

if rank == 0:
    sbuf = np.empty(40000000)
    win = MPI.Win.Create(sbuf, comm=comm)
    win.Lock(0, MPI.LOCK_EXCLUSIVE)
    sbuf[0] = 0
    sbuf[1:4] = np.random.rand(3)
    win.Unlock(0)
    sleep(100000000)
    print("[%02d] Original data %s" % (rank, sbuf))
else:
    rbuf = np.empty(40000000)
    win = MPI.Win.Create(None, comm=comm)
    sleep(1000000)
    win.Lock(0, MPI.LOCK_EXCLUSIVE)
    win.Get(rbuf, 0)
    win.Unlock(0)
    print("[%02d] Received data %s" % (rank, rbuf))

t1 = time.perf_counter()

win.Free()

print(f'Rank: {rank} -- Time: {t1-t0}')
