import numpy as np


class BlockDecomposition(object):
    """
    Class decomposing a cartesian space domain (1D to 3D) into a given number of processors.

    Parameters
    ----------
    nProcs : int
        Total number of processors for space block decomposition.
    gridSizes : list[int]
        Number of grid points in each dimension
    algo : str, optional
        Algorithm used for the block decomposition :

        - Hybrid : approach minimizing interface communication, inspired from
          the `[Hybrid CFD solver] <https://web.stanford.edu/group/ctr/ResBriefs07/5_larsson1_pp47_58.pdf>`_.
        - ChatGPT : quickly generated using `[ChatGPT] <https://chatgpt.com>`_.
        The default is "Hybrid".
    gRank : int, optional
        If provided, the global rank that will determine the local block distribution. Default is None.
    order : str, optional
        The order used when computing the rank block distribution. Default is `C`.
    """

    def __init__(self, nProcs, gridSizes, algo="Hybrid", gRank=None, order="C"):
        dim = len(gridSizes)
        assert dim in [1, 2, 3], "block decomposition only works for 1D, 2D or 3D domains"

        if algo == "ChatGPT":

            nBlocks = [1] * dim
            for i in range(2, int(nProcs**0.5) + 1):
                while nProcs % i == 0:
                    nBlocks[0] *= i
                    nProcs //= i
                    nBlocks.sort()

            if nProcs > 1:
                nBlocks[0] *= nProcs

            nBlocks.sort()
            while len(nBlocks) < dim:
                smallest = nBlocks.pop(0)
                nBlocks += [1, smallest]
                nBlocks.sort()

            while len(nBlocks) > dim:
                smallest = nBlocks.pop(0)
                next_smallest = nBlocks.pop(0)
                nBlocks.append(smallest * next_smallest)
                nBlocks.sort()

        elif algo == "Hybrid":
            rest = nProcs
            facs = {
                1: [1],
                2: [2, 1],
                3: [2, 3, 1],
            }[dim]
            exps = [0] * dim
            for n in range(dim - 1):
                while (rest % facs[n]) == 0:
                    exps[n] = exps[n] + 1
                    rest = rest // facs[n]
            if rest > 1:
                facs[dim - 1] = rest
                exps[dim - 1] = 1

            nBlocks = [1] * dim
            for n in range(dim - 1, -1, -1):
                while exps[n] > 0:
                    dummymax = -1
                    dmax = 0
                    for d, nPts in enumerate(gridSizes):
                        dummy = (nPts + nBlocks[d] - 1) // nBlocks[d]
                        if dummy >= dummymax:
                            dummymax = dummy
                            dmax = d
                    nBlocks[dmax] = nBlocks[dmax] * facs[n]
                    exps[n] = exps[n] - 1

        else:
            raise NotImplementedError(f"algo={algo}")

        # Store attributes
        self.dim = dim
        self.nBlocks = nBlocks
        self.gridSizes = gridSizes

        # Used for rank block distribution
        self.gRank = gRank
        self.order = order

    @property
    def ranks(self):
        assert self.gRank is not None, "gRank attribute needs to be set"
        cart = np.arange(np.prod(self.nBlocks)).reshape(self.nBlocks, order=self.order)
        return list(np.argwhere(cart == self.gRank)[0])

    @property
    def localBounds(self):
        iLocList, nLocList = [], []
        for rank, nPoints, nBlocks in zip(self.ranks, self.gridSizes, self.nBlocks):
            n0 = nPoints // nBlocks
            nRest = nPoints - nBlocks * n0
            nLoc = n0 + 1 * (rank < nRest)
            iLoc = rank * n0 + nRest * (rank >= nRest) + rank * (rank < nRest)

            iLocList.append(iLoc)
            nLocList.append(nLoc)
        return iLocList, nLocList


if __name__ == "__main__":
    # Base usage of this module for a 2D decomposition
    from mpi4py import MPI
    from time import sleep

    comm: MPI.Intracomm = MPI.COMM_WORLD
    MPI_SIZE = comm.Get_size()
    MPI_RANK = comm.Get_rank()

    blocks = BlockDecomposition(MPI_SIZE, [256, 64], gRank=MPI_RANK)
    if MPI_RANK == 0:
        print(f"nBlocks : {blocks.nBlocks}")

    ranks = blocks.ranks
    bounds = blocks.localBounds

    comm.Barrier()
    sleep(0.01 * MPI_RANK)
    print(f"[Rank {MPI_RANK}] pRankX={ranks}, bounds={bounds}")
