from mpi4py import MPI
import numpy as np
import time

def sleep(n):
    tmp = np.random.rand(n)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

if rank == 0:
    sbuf = np.empty(4)
    win = MPI.Win.Create(sbuf, comm=comm)
else:
    rbuf = np.empty(4)
    win = MPI.Win.Create(None, comm=comm)
    # tmp = np.random.rand(int(10000000/2))

group = win.Get_group()

t0 = time.perf_counter()

if rank == 0:
    sleep(10000000)
    # tmp = np.random.rand(100000000)
    for i in range(3):
        if i > 0:
            sleep(100000000)
            win.Wait()
        sbuf[0] = i
        sbuf[1:] = np.random.rand(3)
        print("[%02d] Original data %s" % (rank, sbuf))
        win.Post(group.Incl([1]))
    win.Wait()
else:
    # tmp = np.random.rand(10000)
    # tmp = np.random.rand(10000000)
    # tmp = np.random.rand(1)
    for i in range(3):
        win.Start(group.Excl([1]))
        win.Get(rbuf, 0)
        win.Complete()
        sleep(70000000)
        print("[%02d] Received data %s" % (rank, rbuf))

t1 = time.perf_counter()
group.Free()
win.Free()

print(f'Rank: {rank} -- Time: {t1-t0}')