# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown]
# # Part B: Odd temporal distribution
#
# Accidentally, the numbers of parallel processes used in [Part A](A_run_non_MPI_controller) are always divisors of
# the number of steps. Yet, this does not need to be the case. All controllers can handle odd distributions, e.g. too
# few or too many processes for the steps, or for the last block. We run the multi-level setup of Part A again, with
# 3, 5, 7 and 9 processes for the 8 time steps.

# %%
import matplotlib.pyplot as plt

from pySDC.tutorial.step_6.pfasst_setup import run_pfasst

iterations = run_pfasst(num_proc_list=[3, 5, 7, 9], fname='step_6_B_out.txt', multi_level=True)

# %% tags=["hide-input"]
fig, ax = plt.subplots(figsize=(8, 2.2), constrained_layout=True)
image = ax.imshow(list(iterations.values()), cmap='viridis', aspect='auto', vmin=0)
ax.set_yticks(range(len(iterations)))
ax.set_yticklabels(list(iterations))
ax.set_ylabel('processes')
ax.set_xlabel('time step')
fig.colorbar(image, label='iterations')

# %% [markdown]
# With 3 processes, the 8 steps come in blocks of 3, 3 and 2, and with 9 processes, one process has nothing to do.
# The controllers check which steps are currently active, and only those compute the next block.
#
# :::{admonition} Important things to note
# - This capability becomes useful with adaptive time stepping, where the number of steps is not known in advance.
# - It also works for SDC and MLSDC, where with varying time-step sizes the overall number of steps is not given at
#   the beginning either.
# :::
#
# The check the tests run is inside `run_pfasst`: no step needs more than 8 iterations.
