from mpi4py import MPI
import numpy as np
import time
import threading
from argparse import ArgumentParser


def sleep(n):
    tmp = np.random.rand(n)


def recv(rbuf, source, comm):
    comm.Recv(rbuf[:], source=source, tag=99)


# def send(sbuf, comm):
#     comm.Send(sbuf[:], dest=1, tag=99)


def isend(sbuf, dest, comm):
    return comm.Isend(sbuf[:], dest=dest, tag=99)
    # req.Wait()


def wait(req):
    req.Wait()


def send_stuff(th, space_rank, time_rank, time_comm):
    sbuf = np.empty(40000000)
    sbuf[0] = space_rank
    sbuf[1:4] = np.random.rand(3)
    req = isend(sbuf, time_rank + 1, time_comm)
    th[0] = threading.Thread(target=wait, name='Wait-Thread', args=(req,))
    th[0].start()
    print(f"{time_rank}/{space_rank} - Original data: {sbuf[0:4]}")


def recv_stuff(space_rank, time_rank, time_comm):
    rbuf = np.empty(40000000)
    sleep(10000000 * time_rank)
    recv(rbuf, time_rank - 1, time_comm)
    print(f"{time_rank}/{space_rank} - Received data: {rbuf[0:4]}")


def main(nprocs_space=None):

    # print(MPI.Query_thread(), MPI.THREAD_MULTIPLE)

    # set MPI communicator
    comm = MPI.COMM_WORLD

    world_rank = comm.Get_rank()
    world_size = comm.Get_size()

    # split world communicator to create space-communicators
    color = int(world_rank / nprocs_space)
    space_comm = comm.Split(color=color)
    space_rank = space_comm.Get_rank()

    # split world communicator to create time-communicators
    color = int(world_rank % nprocs_space)
    time_comm = comm.Split(color=color)
    time_rank = time_comm.Get_rank()
    time_size = time_comm.Get_size()

    th = [None]

    comm.Barrier()

    t0 = time.perf_counter()

    if time_rank < time_size - 1:
        send_stuff(th, space_rank, time_rank, time_comm)

    if time_rank > 0:
        recv_stuff(space_rank, time_rank, time_comm)

    if time_rank < time_size - 1:
        sleep(100000000)
        th[0].join()

    t1 = time.perf_counter()

    comm.Barrier()

    maxtime = space_comm.allreduce(t1 - t0, MPI.MAX)

    if space_rank == 0:
        print(f'Time-Rank: {time_rank} -- Time: {maxtime}')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-n", "--nprocs_space", help='Specifies the number of processors in space', type=int)
    args = parser.parse_args()
    main(nprocs_space=args.nprocs_space)
