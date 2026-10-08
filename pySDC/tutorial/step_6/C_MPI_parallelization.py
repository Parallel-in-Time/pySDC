# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#   kernelspec:
#     display_name: Python 3
#     name: python3
#   language_info:
#     name: python
# ---

# %% [markdown]
# # Part C: MPI parallelization
#
# Since PFASST is actually a parallel algorithm, executing it in parallel, e.g. with MPI, might be an interesting
# exercise. For this, pySDC comes with the MPI-parallel controller `controller_MPI`. It is supposed to yield the same
# results as its non-MPI counterpart, and this is what we demonstrate here, for one particular example: the code is the
# same as in [Parts A](A_run_non_MPI_controller) and [B](B_odd_temporal_distribution), with the parameters from
# `pfasst_setup.py`, but with `controller_MPI` instead of the non-MPI controller.
#
# Run it as you would run any MPI program, with one rank per parallel step:
#
# ```bash
# mpirun -np 4 python C_MPI_parallelization.py
# ```
#
# The number of parallel steps is simply the size of `MPI.COMM_WORLD`, so there is nothing to configure: run it on 4
# ranks for 4 parallel steps, on 3 for 3, and so on.

# %%
from pathlib import Path

from mpi4py import MPI

from pySDC.helpers.stats_helper import get_sorted
from pySDC.implementations.controller_classes.controller_MPI import controller_MPI
from pySDC.tutorial.step_6.pfasst_setup import set_parameters_ml

# set MPI communicator
comm = MPI.COMM_WORLD

# get the parameters of Parts A and B
description, controller_params, t0, Tend = set_parameters_ml()

# instantiate the controller, which holds just one step: the one of this rank
controller = controller_MPI(controller_params=controller_params, description=description, comm=comm)

# get initial values on finest level
P = controller.S.levels[0].prob
uinit = P.u_exact(t0)

# call main functions to get things done...
uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)

# %% [markdown]
# Each rank only has the statistics of its own steps, so they are gathered on rank 0, which prints them, in the same
# format as Parts A and B, and writes them to a file for the comparison.

# %%
# filter statistics by type (number of iterations)
iter_counts = get_sorted(stats, type='niter', sortby='time')

# combine statistics into list of statistics
iter_counts_list = comm.gather(iter_counts, root=0)

rank = comm.Get_rank()
size = comm.Get_size()

if rank == 0:
    Path("data").mkdir(parents=True, exist_ok=True)
    f = open(f'data/step_6_C_np{size}.txt', 'w')
    out = 'Working with %2i processes...' % size
    f.write(out + '\n')
    print(out)

    # compute exact solutions and compare with both results
    uex = P.u_exact(Tend)
    err = abs(uex - uend)

    out = 'Error vs. exact solution: %12.8e' % err
    f.write(out + '\n')
    print(out)

    # build one list of statistics instead of list of lists, the sort by time
    iter_counts_gather = [item for sublist in iter_counts_list for item in sublist]
    iter_counts = sorted(iter_counts_gather, key=lambda tup: tup[0])

    # compute and print statistics
    for item in iter_counts:
        out = 'Number of iterations for time %4.2f: %1i ' % (item[0], item[1])
        f.write(out + '\n')
        print(out)

    f.write('\n')
    print()
    f.close()

    assert all(item[1] <= 8 for item in iter_counts), "ERROR: weird iteration counts, got %s" % iter_counts

# %% [markdown]
# ## Results
#
# Our CI runs this on 1, 2, 4 and 8 ranks, as in Part A, and on 3, 5, 7 and 9 ranks, as in Part B, and checks that
# the MPI controller needs the same iterations and reaches the same accuracy as the non-MPI one, up to machine
# precision. These are the results of the CI run that built this page, first for the even distributions:
#
# :::{literalinclude} /../../data/step_6_C1_out.txt
# :language: text
# :::
#
# And for the odd ones:
#
# :::{literalinclude} /../../data/step_6_C2_out.txt
# :language: text
# :::
#
# :::{admonition} Important things to note
# - This example also shows how the statistics of multiple MPI processes can be gathered and processed by rank 0.
# - The controller needs a working installation of `mpi4py`. Since this is not always easy to achieve, and since
#   debugging a parallel program can cause a lot of headaches, the non-MPI controller performs the same operations in
#   serial.
# - The test that covers this part runs the same file on 1, 2, 3, 4, 5, 7, 8 and 9 ranks through
#   [mpi-pytest](https://github.com/firedrakeproject/mpi-pytest), which is also how the rest of pySDC's MPI tests are
#   run.
# :::
