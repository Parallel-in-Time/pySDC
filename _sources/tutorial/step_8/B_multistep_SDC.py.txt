# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown]
# # Part B: Multi-step SDC
#
# What happens if we want parallel time steps, but only a single level? The result is **multi-step SDC**
# (MSSDC): after each sweep, a step sends its result forward, and the next step picks it up in its next iteration,
# not in the current one. This is like doing only the smoother of a multigrid method. Parallelization is dead
# simple and needs no coarsening. Yet, without the stabilization of a coarse level, the iteration counts grow
# significantly when more steps are computed in parallel. The information can also be picked up immediately, but
# then the steps wait for each other and the method is not parallel any more.
#
# The controller parameter `mssdc_jac` chooses between the two: `True` for the "parallel", Jacobi-like variant,
# `False` for the "serial", Gauss-Seidel-like one. We compare both with PFASST, with 8 parallel steps.

# %%
import os
from pathlib import Path

import matplotlib.pyplot as plt

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
    'bc': 'dirichlet-zero',  # boundary conditions
}

# initialize step parameters
step_params = {'maxiter': 50}

# initialize space transfer parameters
space_transfer_params = {'rorder': 2, 'iorder': 6}

# initialize controller parameters
controller_params = {'logger_level': 40}

# fill description dictionary for easy step instantiation
description = {
    'problem_class': heatNd_unforced,
    'sweeper_class': generic_implicit,
    'sweeper_params': sweeper_params,
    'level_params': level_params,
    'step_params': step_params,
    'space_transfer_class': mesh_to_mesh,
    'space_transfer_params': space_transfer_params,
}

# set up parameters for PFASST run
problem_params['nvars'] = [63, 31]
description['problem_params'] = problem_params.copy()
description_pfasst = description.copy()

# set up parameters for MSSDC run
problem_params['nvars'] = [63]
description['problem_params'] = problem_params.copy()
description_mssdc = description.copy()

controller_params['mssdc_jac'] = True
controller_params_jac = controller_params.copy()
controller_params['mssdc_jac'] = False
controller_params_gs = controller_params.copy()

# %% [markdown]
# The multi-step SDC descriptions are copies of the PFASST one, down to the transfer class, which a single level
# does not need. pySDC warns about that, and the `logger_level` of 40 silences these (safe) warnings.
#
# ## Three runs

# %%
# set time parameters
t0 = 0.0
Tend = 1.0

# set up list of parallel time-steps to run PFASST/MSSDC with
num_proc = 8

# instantiate controllers
controller_mssdc_jac = controller_nonMPI(
    num_procs=num_proc, controller_params=controller_params_jac, description=description_mssdc
)
controller_mssdc_gs = controller_nonMPI(
    num_procs=num_proc, controller_params=controller_params_gs, description=description_mssdc
)
controller_pfasst = controller_nonMPI(
    num_procs=num_proc, controller_params=controller_params, description=description_pfasst
)

# get initial values on finest level
P = controller_mssdc_jac.MS[0].levels[0].prob
uinit = P.u_exact(t0)

# call main functions to get things done...
uend_pfasst, stats_pfasst = controller_pfasst.run(u0=uinit, t0=t0, Tend=Tend)
uend_mssdc_jac, stats_mssdc_jac = controller_mssdc_jac.run(u0=uinit, t0=t0, Tend=Tend)
uend_mssdc_gs, stats_mssdc_gs = controller_mssdc_gs.run(u0=uinit, t0=t0, Tend=Tend)

# compute exact solution and compare for both runs
uex = P.u_exact(Tend)
err_mssdc_jac = abs(uex - uend_mssdc_jac)
err_mssdc_gs = abs(uex - uend_mssdc_gs)
err_pfasst = abs(uex - uend_pfasst)
diff_jac = abs(uend_mssdc_jac - uend_pfasst)
diff_gs = abs(uend_mssdc_gs - uend_pfasst)
diff_jac_gs = abs(uend_mssdc_gs - uend_mssdc_jac)

print('Error PFASST: %12.8e' % err_pfasst)
print('Error parallel MSSDC: %12.8e' % err_mssdc_jac)
print('Error serial MSSDC: %12.8e' % err_mssdc_gs)
print('Diff PFASST vs. parallel MSSDC: %12.8e' % diff_jac)
print('Diff PFASST vs. serial MSSDC: %12.8e' % diff_gs)
print('Diff parallel vs. serial MSSDC: %12.8e' % diff_jac_gs)

# %% [markdown]
# All three arrive at the same solution. The difference is in how long it takes:

# %%
# convert filtered statistics to list of iterations count, sorted by process
iter_counts_pfasst = get_sorted(stats_pfasst, type='niter', sortby='time')
iter_counts_mssdc_jac = get_sorted(stats_mssdc_jac, type='niter', sortby='time')
iter_counts_mssdc_gs = get_sorted(stats_mssdc_gs, type='niter', sortby='time')

# compute and print statistics
for item_pfasst, item_mssdc_jac, item_mssdc_gs in zip(
    iter_counts_pfasst, iter_counts_mssdc_jac, iter_counts_mssdc_gs, strict=True
):
    print(
        'Number of iterations for time %4.2f (PFASST/parMSSDC/serMSSDC): %2i / %2i / %2i'
        % (item_pfasst[0], item_pfasst[1], item_mssdc_jac[1], item_mssdc_gs[1])
    )

# %% tags=["hide-input"]
fig, ax = plt.subplots(figsize=(6, 3.5))
for name, counts in [
    ('PFASST', iter_counts_pfasst),
    ('parallel MSSDC', iter_counts_mssdc_jac),
    ('serial MSSDC', iter_counts_mssdc_gs),
]:
    ax.plot(range(len(counts)), [n for _, n in counts], 'o-', label=name)
ax.set_xlabel('process (time step)')
ax.set_ylabel('iterations')
ax.legend(frameon=False)
ax.grid(alpha=0.3)
fig.tight_layout()

# %% [markdown]
# Parallel multi-step SDC needs more and more iterations along the block, up to more than twice as many as PFASST.
# The serial variant needs fewer, but its steps wait for each other. The residuals, with the helper
# `show_residual_across_simulation` from `pySDC.helpers.visualization_tools`, for the parallel and then the serial
# variant, over the processes (rows) and iterations (columns):

# %%
Path("data").mkdir(parents=True, exist_ok=True)
# call helper routine to produce residual plot
fig_jac = show_residual_across_simulation(stats_mssdc_jac, 'data/step_8_residuals_mssdc_jac.png')
fig_gs = show_residual_across_simulation(stats_mssdc_gs, 'data/step_8_residuals_mssdc_gs.png')

# %% [markdown]
# The checks the tests run:

# %%
assert os.path.isfile('data/step_8_residuals_mssdc_jac.png')
assert os.path.isfile('data/step_8_residuals_mssdc_gs.png')
assert (
    diff_jac < 3.1e-10
), f"ERROR: difference between PFASST and parallel MSSDC controller is too large, got {diff_jac}"
assert diff_gs < 3.1e-10, f"ERROR: difference between PFASST and serial MSSDC controller is too large, got {diff_gs}"
assert (
    diff_jac_gs < 3.1e-10
), f"ERROR: difference between parallel and serial MSSDC controller is too large, got {diff_jac_gs}"
