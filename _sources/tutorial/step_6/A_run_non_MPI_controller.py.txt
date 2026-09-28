# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown]
# # Part A: The non-MPI controller
#
# pySDC comes with (at least) two controllers: the standard, non-MPI controller we have used so far, and the
# MPI-parallel one. The non-MPI controller runs simulations without having to worry about parallelization and MPI
# installations. By monitoring the convergence, it can already give a detailed idea of how PFASST will work for a
# given problem.
#
# We run an unforced heat equation, on a smaller grid than in [Step 5](../step_5/B_my_first_PFASST_run), once on a
# single level, as SDC, and then with two levels on 1, 2, 4 and 8 processes: MLSDC on one, PFASST on more. The setup is in `pfasst_setup.py` next to this tutorial, as
# [Parts B](B_odd_temporal_distribution) and [C](C_MPI_parallelization) use it as well. The parameters of the
# multi-level runs:
#
# :::{literalinclude} pfasst_setup.py
# :pyobject: set_parameters_ml
# :::
#
# The single-level ones, `set_parameters_sl`, are the same problem and sweeper on one level, without the transfer
# class and without the controller options above. And this is the loop that runs the controller for each number of
# processes, printing the error and the iterations of each step:
#
# :::{literalinclude} pfasst_setup.py
# :pyobject: run_pfasst
# :::
#
# The output also goes to a file in `data/`, for the comparison with the MPI controller in Part C.

# %%
import matplotlib.pyplot as plt

from pySDC.tutorial.step_6.pfasst_setup import run_pfasst

iterations_sl = run_pfasst(num_proc_list=[1], fname='step_6_A_sl_out.txt', multi_level=False)

# %% [markdown]
# And with two levels, PFASST:

# %%
iterations_ml = run_pfasst(num_proc_list=[1, 2, 4, 8], fname='step_6_A_ml_out.txt', multi_level=True)

# %% tags=["hide-input"]
fig, ax = plt.subplots(figsize=(8, 2.4), constrained_layout=True)
rows = {'SDC, 1': iterations_sl[1]}
rows.update({f'{"MLSDC" if n == 1 else "PFASST"}, {n}': counts for n, counts in iterations_ml.items()})
image = ax.imshow(list(rows.values()), cmap='viridis', aspect='auto', vmin=0)
ax.set_yticks(range(len(rows)))
ax.set_yticklabels(list(rows))
ax.set_ylabel('method, processes')
ax.set_xlabel('time step')
fig.colorbar(image, label='iterations')

# %% [markdown]
# :::{admonition} Important things to note
# - If you don't want to deal with parallelization and/or are only interested in SDC, MLSDC or the convergence of
#   PFASST, use the non-MPI controller.
# - If you care about parallelization, use the MPI controller, see [Part C](C_MPI_parallelization).
# :::
#
# The check the tests run is inside `run_pfasst`: no step needs more than 8 iterations.
