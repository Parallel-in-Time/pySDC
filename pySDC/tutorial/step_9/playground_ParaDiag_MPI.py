import sys
from pathlib import Path

from mpi4py import MPI
import numpy as np

from pySDC.helpers.stats_helper import get_sorted
from pySDC.implementations.controller_classes.controller_ParaDiag_MPI import controller_ParaDiag_MPI
from pySDC.tutorial.step_9.D_paradiag_MPI import (
    get_description,
    get_controller_params,
    format_result,
    num_steps_total,
)

if __name__ == "__main__":
    """
    A simple test program to do MPI-parallel ParaDiag runs

    One time-step per rank, so the communicator spans the block that ParaDiag diagonalizes across. We
    always integrate `num_steps_total` steps, so with fewer ranks the controller simply windows
    through more blocks.
    """

    # set MPI communicator
    comm = MPI.COMM_WORLD

    # one step per rank, so the number of ranks is the block size
    block_size = comm.size

    # instantiate the controller
    controller = controller_ParaDiag_MPI(
        controller_params=get_controller_params(), description=get_description(), comm=comm
    )

    # ParaDiag diagonalizes in time, so the solution becomes complex
    P = controller.S.levels[0].prob
    P.init = tuple([*P.init[:2]] + [np.dtype('complex128')])

    # get initial values
    t0 = 0.0
    dt = controller.S.levels[0].params.dt
    Tend = num_steps_total * dt
    uinit = P.u_exact(t0)

    # call main function to get things done...
    uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)

    # gathering the iteration counts is collective, so every rank has to take part
    niter = [int(me[1]) for me in get_sorted(stats, type='niter', sortby='time', comm=comm)]

    # only the last rank has the end point of the block, so only it writes the output
    if comm.rank == comm.size - 1:
        fname = sys.argv[1] if len(sys.argv) == 2 else 'step_9_D_out.txt'
        Path("data").mkdir(parents=True, exist_ok=True)
        f = open('data/' + fname, 'a')
        out = format_result('MPI', block_size, niter, abs(uend - P.u_exact(Tend)))
        f.write(out + '\n')
        print(out)
        f.close()
