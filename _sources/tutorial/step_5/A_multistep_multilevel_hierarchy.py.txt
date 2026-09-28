# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown]
# # Part A: Multistep multilevel hierarchy
#
# For SDC and MLSDC, one `step` was all we needed. PFASST works on several time steps at once, and a controller
# represents them simply as a list of steps, its attribute `MS`. The nice thing about going from MLSDC to PFASST is
# that only one number changes: `num_procs`, the number of time steps the controller works on at once. This
# controller only emulates working on them in parallel, see the box below; running them really in parallel takes the
# MPI controller as well, with one process per step.

# %%
import matplotlib.pyplot as plt

from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.implementations.problem_classes.HeatEquation_ND_FD import heatNd_forced
from pySDC.implementations.sweeper_classes.imex_1st_order import imex_1st_order
from pySDC.implementations.transfer_classes.TransferMesh import mesh_to_mesh

# initialize level parameters
level_params = {'restol': 1e-10, 'dt': 0.5}

# initialize sweeper parameters
sweeper_params = {'quad_type': 'RADAU-RIGHT', 'num_nodes': [3]}

# initialize problem parameters
problem_params = {
    'nu': 0.1,  # diffusion coefficient
    'freq': 4,  # frequency for the test value
    'nvars': [31, 15, 7],  # number of degrees of freedom for each level
    'bc': 'dirichlet-zero',  # boundary conditions
}

# initialize step parameters
step_params = {'maxiter': 20}

# initialize space transfer parameters
space_transfer_params = {'rorder': 2, 'iorder': 6}

# fill description dictionary for easy step instantiation
description = {
    'problem_class': heatNd_forced,  # pass problem class
    'problem_params': problem_params,  # pass problem parameters
    'sweeper_class': imex_1st_order,  # pass sweeper (see part B)
    'sweeper_params': sweeper_params,  # pass sweeper parameters
    'level_params': level_params,  # pass level parameters
    'step_params': step_params,  # pass step parameters
    'space_transfer_class': mesh_to_mesh,  # pass spatial transfer class
    'space_transfer_params': space_transfer_params,  # pass parameters for spatial transfer
}

# %% [markdown]
# The description is the one of an MLSDC run with three levels in space. The controller gets 10 processes:

# %% tags=["scroll-output"]
# instantiate controller
controller = controller_nonMPI(num_procs=10, controller_params={}, description=description)

# %%
# check number of levels
for i in range(len(controller.MS)):
    print("Process %2i has %2i levels" % (i, len(controller.MS[i].levels)))

# %% [markdown]
# Ten steps, each a full hierarchy of its own, with its own levels, problems and sweepers:

# %% tags=["hide-input"]
fig, ax = plt.subplots(figsize=(8, 2.6))
for i, S in enumerate(controller.MS):
    for L in S.levels:
        ax.add_patch(
            plt.Rectangle((i + 0.05, L.level_index + 0.05), 0.9, 0.9, color=plt.cm.Blues(0.8 - 0.25 * L.level_index))
        )
        ax.text(i + 0.5, L.level_index + 0.5, L.prob.nvars[0], ha='center', va='center', color='white', fontsize=9)
ax.set_xlim(0, len(controller.MS))
ax.set_ylim(len(controller.MS[0].levels), 0)
ax.set_xticks([i + 0.5 for i in range(len(controller.MS))])
ax.set_xticklabels([f'step {i}' for i in range(len(controller.MS))], fontsize=8)
ax.set_yticks([0.5, 1.5, 2.5])
ax.set_yticklabels(['level 0 (fine)', 'level 1', 'level 2 (coarse)'], fontsize=8)
ax.set_title('the steps in controller.MS, with the number of unknowns on each level', fontsize=10)
ax.set_frame_on(False)
fig.tight_layout()

# %%
print('Step 0 and step 1 share their problem:', controller.MS[0].levels[0].prob is controller.MS[1].levels[0].prob)

# %% [markdown]
# :::{admonition} Important things to note
# - Controllers with the `_nonMPI` suffix only emulate parallelism, which saves the tedious installation of mpi4py
#   and gives full access to all data at all times. The algorithm is the same, but the steps are computed
#   serially: the controller moves all of them through the algorithm together, one stage at a time, and within
#   each stage it handles one step after another. The `MPI` controllers run them really in parallel and should give
#   the same results, see [Step 6](../step_6).
# - All steps of a controller are created from the same description, so they all have the same levels, as the
#   check below confirms.
# :::
#
# The check the tests run:

# %%
assert all(len(S.levels) == 3 for S in controller.MS), "ERROR: not all steps have the same number of levels"
