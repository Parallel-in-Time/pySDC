"""
This script shows how to run ParaDiag with actual MPI parallelism across the time-steps.

Parts A to D all used the "virtually parallel" controller, which holds every step in one process and
is what you want while developing. Here we use ``controller_ParaDiag_MPI`` instead: one time-step per
rank, with the communicator spanning the block that ParaDiag diagonalizes across.

The point is that nothing about the *method* changes. The description, the controller parameters and
the alpha settings are the ones Part D already used -- ``run`` there takes a communicator, and passing
one is the entire difference. So you can develop a setup serially and then run it in parallel without
touching it. Run this the way you would run any MPI program::

    mpirun -np 4 python E_paradiag_MPI.py

We always integrate the same total number of time-steps and only vary how many of them run in
parallel, so the number of ranks is the block size: four steps in total with a block size of one, two
or four means windowing through four, two or one block respectively.

What that does and does not leave unchanged is worth being precise about:

- Windowing does not change what is being solved. Where the iteration count comes out the same, the
  answers are identical to the last bit.
- The iteration count usually does come out the same, but it need not: a fixed alpha of 1e-2 is loose
  enough that a block of two or four steps costs one extra iteration over a block of one, while 1e-4
  and 1e-8 do not notice. Where that happens the two runs stop at slightly different residuals, so
  their errors differ by ~2e-8 -- three orders inside the discretisation error of ~3e-5, and leftover
  iteration error rather than a different solution.
- Adaptive alpha does not notice the block size either, but for a more interesting reason: it picks a
  *different* alpha for each one, because gamma scales with the number of steps in the block, and
  still converges in the same number of iterations. That is the margin the adaptive strategy buys
  you, made visible.
- At a given block size the MPI and the virtually parallel controllers have to agree exactly. That is
  the comparison against Part D.

Two properties of ParaDiag are worth keeping in mind when going parallel, because they differ from
PFASST:

- All steps of a block have to iterate together. In PFASST an early step can converge and drop out of
  the iteration, which is what makes it pipelined. ParaDiag cannot do that: the transform in time
  needs every step, so a step that stopped early would leave the others waiting forever.
- The block is therefore always full. If the end time does not divide into whole blocks, ParaDiag
  solves past it rather than truncating, and says so.
"""

import sys
from pathlib import Path

from mpi4py import MPI

from pySDC.tutorial.step_9.D_adaptive_alpha import alpha_settings, format_result, run


def main(fname='step_9_E_out.txt'):
    """
    Run every alpha setting on this communicator, one time-step per rank.

    The block size is simply ``MPI.COMM_WORLD.size``, so there is nothing to configure: run it on one
    rank and ParaDiag windows through four blocks of one step, run it on four and it does one block
    of four.

    Args:
        fname (str): file under ``data/`` to append the results to
    """

    comm = MPI.COMM_WORLD

    lines = []
    for alpha in alpha_settings:
        uend, niter, error, final_alpha = run(alpha, comm.size, comm=comm)
        lines.append(format_result('MPI', alpha, niter, error, final_alpha))

    # only the last rank holds the end point of the block, so only it writes the output
    if comm.rank == comm.size - 1:
        Path("data").mkdir(parents=True, exist_ok=True)
        with open('data/' + fname, 'a') as f:
            for line in lines:
                f.write(line + '\n')
                print(line)


if __name__ == "__main__":
    # Run it as you would any MPI program, one rank per time-step:
    #     mpirun -np 4 python E_paradiag_MPI.py
    main(sys.argv[1] if len(sys.argv) == 2 else 'step_9_E_out.txt')
