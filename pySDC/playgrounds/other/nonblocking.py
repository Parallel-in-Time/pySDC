from mpi4py import MPI
import numpy as np

def main():

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    data = None
    if rank == 0:
        data = np.array([1, 2, 3])
        req = comm.isend(data, dest=1)
        req.wait()
    elif rank == 1:
        data = comm.recv(source=0)
        print(data)
    #
    # if rank == 0:
    #     n = 10
    #     for i in range(n):
    #         tmp = np.array(i, dtype=int)
    #         # print(f'{rank} sending {i} to {1} - START')
    #         comm.send(tmp, dest=1)
    #         # req = comm.isend(tmp, dest=1, tag=i)
    #         # req = comm.Isend((tmp, MPI.INT), dest=1, tag=i)
    #         # req.wait()
    #         # print(f'{rank} sending {i} to {1} - STOP')
    # elif rank == 1:
    #     pass

if __name__ == '__main__':
    main()
