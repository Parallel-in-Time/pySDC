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
# # Part E: MPI-parallel ParaDiag
#
# Parts A to D all ran ParaDiag with the "virtually parallel" controller, which keeps every step in a single process.
# That is what you want while developing, but it does not actually run in parallel. This part uses
# `controller_ParaDiag_MPI` instead, with one time step per rank and the communicator spanning the block that is
# diagonalized.
#
# Nothing about the method changes: the description, the controller parameters and the $\alpha$ settings are the
# ones [Part D](D_adaptive_alpha) already used. In fact `run` there takes a communicator, and passing one is the
# entire difference, so you can develop a setup serially and then run it in parallel without touching it.
#
# Run it the way you would run any MPI program, with one rank per time step:
#
# ```bash
# mpirun -np 4 python E_paradiag_MPI.py
# ```
#
# We always integrate the same total number of time steps and only vary how many of them run in parallel, so the
# number of ranks is the block size. With four steps in total and a block size of one, two or four, the controller
# windows through four, two or one block respectively.

# %%
from pathlib import Path

from mpi4py import MPI

from pySDC.tutorial.step_9.paradiag_setup import alpha_settings, format_result, run

comm = MPI.COMM_WORLD

lines = []
for alpha in alpha_settings:
    uend, niter, error, final_alpha = run(alpha, comm.size, comm=comm)
    # the block size goes in the label, so that the results of all block sizes can be put side by side
    lines.append(format_result(f'MPI on {comm.size}', alpha, niter, error, final_alpha))

# only the last rank holds the end point of the block, so only it writes the output
if comm.rank == comm.size - 1:
    Path("data").mkdir(parents=True, exist_ok=True)
    with open(f'data/step_9_E_np{comm.size}.txt', 'w') as f:
        for line in lines:
            f.write(line + '\n')
            print(line)

# %% [markdown]
# ## Results
#
# Our CI runs this on one, two and four ranks. These are the results of the CI run that built this page:
#
# :::{literalinclude} /../../data/step_9_E_out.txt
# :language: text
# :::
#
# And these are the ones of [Part D](D_adaptive_alpha), with the virtually parallel controller on one block of four
# steps:
#
# :::{literalinclude} /../../data/step_9_D_out.txt
# :language: text
# :::
#
# :::{admonition} Important things to note
# - All steps of a block iterate together. In PFASST an early step can converge and drop out, which is what makes it
#   pipelined; ParaDiag cannot do that, because the transform in time needs every step. A step that stopped early
#   would leave the others waiting.
# - Consequently the block is always full. If the end time does not divide into whole blocks, ParaDiag solves past
#   it rather than truncating, and says so.
# - At a given block size the MPI and the virtually parallel controllers must agree exactly, which is the comparison
#   against Part D.
# - Windowing does not change what is being solved. Where the iteration count comes out the same the answers agree
#   to about $10^{-11}$; where it does not (a fixed $\alpha$ of $10^{-2}$ is loose enough that a larger block costs
#   one extra iteration) the two runs stop at slightly different residuals, and their errors differ by about
#   $2 \cdot 10^{-8}$, three orders inside the discretisation error.
# - Adaptive $\alpha$ does not notice the block size either, but for a more interesting reason: it picks a
#   *different* $\alpha$ for each one, because $\gamma$ scales with the number of steps in the block, and still
#   converges in the same number of iterations.
# :::
