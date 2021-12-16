from mpi4py import MPI
import numpy as np
import time


def sleep(n):
    tmp = np.random.rand(n)


comm = MPI.COMM_WORLD
rank = comm.Get_rank()

comm.Barrier()

t0 = time.perf_counter()

if rank == 0:
    sbuf = np.empty(40000000)
    sbuf[0] = 0
    sbuf[1:4] = np.random.rand(3)
    req = comm.Isend(sbuf[:], dest=1, tag=99)
    sleep(100000000)
    req.wait()
    print("[%02d] Original data %s" % (rank, sbuf))
else:
    rbuf = np.empty(40000000)
    sleep(10000000)
    comm.Recv(rbuf[:], source=0, tag=99)
    print("[%02d] Received data %s" % (rank, rbuf))

t1 = time.perf_counter()

print(f'Rank: {rank} -- Time: {t1-t0}')