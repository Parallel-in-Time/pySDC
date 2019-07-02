from mpi4py import MPI
import numpy as np
import time
import threading
from argparse import ArgumentParser

def sleep(n):
    tmp = np.random.rand(n)


def recv(rbuf, comm):
    comm.Recv(rbuf[:], source=0, tag=99)


# def send(sbuf, comm):
#     comm.Send(sbuf[:], dest=1, tag=99)


def isend(sbuf, comm):
    req = comm.Isend(sbuf[:], dest=1, tag=99)
    req.Wait()


def main(nprocs_space=None):

    # print(MPI.Query_thread(), MPI.THREAD_MULTIPLE)

    # set MPI communicator
    comm = MPI.COMM_WORLD

    world_rank = comm.Get_rank()
    world_size = comm.Get_size()
    assert world_size == 2 * nprocs_space

    # split world communicator to create space-communicators
    color = int(world_rank / nprocs_space)
    space_comm = comm.Split(color=color)
    space_rank = space_comm.Get_rank()

    # split world communicator to create time-communicators
    color = int(world_rank % nprocs_space)
    time_comm = comm.Split(color=color)
    time_rank = time_comm.Get_rank()

    comm.Barrier()

    t0 = time.time()

    if time_rank == 0:
        sbuf = np.empty(40000000)
        sbuf[0] = space_rank
        sbuf[1:4] = np.random.rand(3)
        send_thread = threading.Thread(target=isend, args=(sbuf, time_comm))
        send_thread.start()
        sleep(100000000)
        send_thread.join()
        print(f"{time_rank}/{space_rank} - Original data: {sbuf[0:4]}")
    else:
        rbuf = np.empty(40000000)
        sleep(10000000)
        recv(rbuf, time_comm)
        print(f"{time_rank}/{space_rank} - Received data: {rbuf[0:4]}")

    t1 = time.time()

    comm.Barrier()

    maxtime = space_comm.allreduce(t1 - t0, MPI.MAX)

    if space_rank == 0:
        print(f'Time-Rank: {time_rank} -- Time: {maxtime}')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-n", "--nprocs_space", help='Specifies the number of processors in space', type=int)
    args = parser.parse_args()
    main(nprocs_space=args.nprocs_space)
