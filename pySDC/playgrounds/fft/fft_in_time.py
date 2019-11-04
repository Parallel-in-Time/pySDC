from mpi4py import MPI
from mpi4py_fft import PFFT, newDistArray
from mpi4py_fft.pencil import Subcomm

import numpy as np


def main():

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    L = 16
    N = 1

    pfft = PFFT(comm, [L, N], axes=0, dtype=np.complex128, grid=(0, -1))

    tmp_u = newDistArray(pfft, False)

    # print(rank, tmp_u.shape)
    Lloc = tmp_u.shape[0]
    tvalues = np.linspace(rank * 1.0 / size, (rank + 1) * 1.0 / size, Lloc, endpoint=False)
    print(tvalues)

    u = np.zeros(tmp_u.shape)
    for n in range(N):
        u[:, n] = np.sin((n + 1) * 2 * np.pi * tvalues)
    print(u)

    print(pfft.forward(u))



if __name__ == '__main__':
    main()