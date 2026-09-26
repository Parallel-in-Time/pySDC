# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown]
# # Part A: Visualizing residuals
#
# pySDC comes with a helper that shows how the residuals of a PFASST run evolve, over all processes and all
# iterations: `show_residual_across_simulation` in `pySDC.helpers.visualization_tools`. Using it is (supposed to
# be) simple: hand it the `stats` of a run.
#
# We use the PFASST setup of [Step 6](../step_6), for the heat equation on two levels, with 8 processes.

# %%
import os
from pathlib import Path

from pySDC.helpers.stats_helper import get_sorted
from pySDC.helpers.visualization_tools import show_residual_across_simulation
from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.implementations.problem_classes.HeatEquation_ND_FD import heatNd_unforced
from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit
from pySDC.implementations.transfer_classes.TransferMesh import mesh_to_mesh

# initialize level parameters
level_params = {'restol': 5e-10, 'dt': 0.125}

# initialize sweeper parameters
sweeper_params = {'quad_type': 'RADAU-RIGHT', 'num_nodes': [3], 'QI': 'LU'}

# initialize problem parameters
problem_params = {
    'nu': 0.1,  # diffusion coefficient
    'freq': 2,  # frequency for the test value
    'nvars': [63, 31],  # number of degrees of freedom for each level
    'bc': 'dirichlet-zero',  # boundary conditions
}

# initialize step parameters
step_params = {'maxiter': 50, 'errtol': 1e-05}

# initialize space transfer parameters
space_transfer_params = {'rorder': 2, 'iorder': 6}

# initialize controller parameters
controller_params = {
    'logger_level': 30,
    'all_to_done': True,  # can ask the controller to keep iterating all steps until the end
    'predict_type': 'pfasst_burnin',  # PFASST's coarse-level predictor
}

# fill description dictionary for easy step instantiation
description = {
    'problem_class': heatNd_unforced,
    'problem_params': problem_params,
    'sweeper_class': generic_implicit,
    'sweeper_params': sweeper_params,
    'level_params': level_params,
    'step_params': step_params,
    'space_transfer_class': mesh_to_mesh,
    'space_transfer_params': space_transfer_params,
}

# set time parameters
t0 = 0.0
Tend = 1.0

# use 8 processes here
num_proc = 8

# instantiate controller
controller = controller_nonMPI(num_procs=num_proc, controller_params=controller_params, description=description)

# get initial values on finest level
P = controller.MS[0].levels[0].prob
uinit = P.u_exact(t0)

# call main function to get things done...
uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)

# compute exact solution and compare (for testing purposes only)
uex = P.u_exact(Tend)
err = abs(uex - uend)

# %% [markdown]
# Eight time steps on eight processes: one block of PFASST. Each step took the same number of iterations, because
# `all_to_done` keeps all of them iterating until the last one is done:

# %%
# filter statistics by type (number of iterations)
iter_counts = get_sorted(stats, type='niter', sortby='time')

for item in iter_counts:
    print('Number of iterations for time %4.2f: %1i' % item)
min_iter = min(item[1] for item in iter_counts)
max_iter = max(item[1] for item in iter_counts)

# %% [markdown]
# ## The residuals
#
# One call. The helper saves the figure to the file we name, and returns it too:

# %%
Path("data").mkdir(parents=True, exist_ok=True)
fname = 'data/step_8_residuals.png'
fig = show_residual_across_simulation(stats=stats, fname=fname)

# %% [markdown]
# Each row is a process, i.e. a time step, each column an iteration, and the color is the residual. The first steps
# converge first, and the later ones follow, as the information travels forward through the block.
#
# :::{admonition} Important things to note
# - The helper shows the residuals over all processes and all iterations, but only for a single block of PFASST.
# - It needs nothing but `stats`: the number of processes and iterations comes from the statistics.
# :::
#
# The checks the tests run:

# %%
assert err < 6.1555e-05, f'ERROR: error is too large, got {err}'
assert os.path.isfile(fname), 'ERROR: residual plot has not been created'
assert min_iter == 7 and max_iter == 7, f"ERROR: number of iterations not as expected, got {min_iter} and {max_iter}"
