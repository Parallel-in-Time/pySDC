import mpi4py

mpi4py.rc.threaded = True
mpi4py.rc.thread_level = "multiple"

from mpi4py import MPI
import numpy as np
import time

from multiprocessing import Process


def sleep(n):
    tmp = np.random.rand(n)


def wait(req):
    print('p0', req.Test())
    req.Wait()
    print('p1', req.Test())


def isend(sbuf, comm):
    print('sending')
    comm.send(sbuf, dest=1, tag=99)
    # print('waiting')
    # req.Wait()
    print('done')


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    comm.Barrier()

    t0 = time.perf_counter()

    if rank == 0:
        sbuf = np.empty(4000000)
        sbuf[0] = 0
        sbuf[1:4] = np.random.rand(3)
        p = Process(target=isend, args=(sbuf, comm))
        p.start()
        sleep(100000000)
        p.join()
        print("[%02d] Original data %s" % (rank, sbuf))
    else:
        print('working')
        sleep(1000000)
        print('receiving')
        rbuf = comm.recv(source=0, tag=99)
        print('rdone')
        print("[%02d] Received data %s" % (rank, rbuf))

    t1 = time.perf_counter()

    comm.Barrier()

    print(f'Rank: {rank} -- Time: {t1-t0}')


if __name__ == '__main__':
    main()
