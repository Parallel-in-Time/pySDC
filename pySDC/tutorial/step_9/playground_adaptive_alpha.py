import sys
from pathlib import Path

from mpi4py import MPI

from pySDC.tutorial.step_9.E_adaptive_alpha import alpha_settings, format_result, run

if __name__ == "__main__":
    """
    Compare fixed and adaptive alpha with the MPI-parallel ParaDiag controller.

    One time-step per rank, so the communicator spans the block ParaDiag diagonalizes across. Alpha is
    a property of the method rather than of the parallelization, so these numbers have to match the
    ones the virtually parallel controller produces.
    """

    comm = MPI.COMM_WORLD

    lines = []
    for alpha in alpha_settings:
        uend, niter, error, final_alpha = run(alpha, comm.size, comm=comm)
        lines.append(format_result('MPI', alpha, niter, error, final_alpha))

    # only the last rank has the end point of the block, so only it writes the output
    if comm.rank == comm.size - 1:
        fname = sys.argv[1] if len(sys.argv) == 2 else 'step_9_E_out.txt'
        Path("data").mkdir(parents=True, exist_ok=True)
        with open('data/' + fname, 'a') as f:
            for line in lines:
                f.write(line + '\n')
                print(line)
