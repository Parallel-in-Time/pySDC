# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown]
# # Part B: Multilevel hierarchy
#
# How does a step get several levels? From the description: wherever a problem, sweeper or level parameter is a
# **list** instead of a single value, the step creates one level per entry, the first entry being the finest level.
# There are two generic ways to coarsen:
#
# - in space: make the problem parameter `nvars` a list,
# - in the collocation order: make the sweeper parameter `num_nodes` a list.
#
# A third way is to make an entry of the description itself a list, e.g. a list of problem classes; we use that in
# [Part D](D_MLSDC_with_particles).

# %%
from pySDC.core.step import Step
from pySDC.implementations.problem_classes.HeatEquation_ND_FD import heatNd_unforced
from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit
from pySDC.implementations.transfer_classes.TransferMesh import mesh_to_mesh

# initialize level parameters
level_params = {'restol': 1e-10, 'dt': 0.1}

# initialize sweeper parameters
sweeper_params = {
    'quad_type': 'RADAU-RIGHT',
    'num_nodes': [5, 3],  # number of collocation nodes for each level
    'QI': 'LU',
}

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
space_transfer_params = {'rorder': 2, 'iorder': 2}

# fill description dictionary for easy step instantiation
description = {
    'problem_class': heatNd_unforced,
    'problem_params': problem_params,
    'sweeper_class': generic_implicit,
    'sweeper_params': sweeper_params,
    'level_params': level_params,
    'step_params': step_params,
    'space_transfer_class': mesh_to_mesh,  # the transfer between levels is part of the description, too
    'space_transfer_params': space_transfer_params,
}

# now the description contains more or less everything we need to create a step with multiple levels
S = Step(description=description)

# %% [markdown]
# During the setup, dictionaries with list entries are turned into lists of dictionaries, one for each level:

# %%
for l in range(len(S.levels)):
    L = S.levels[l]
    print('Level %2i: nvars = %4i -- nnodes = %2i' % (l, L.prob.nvars[0], L.sweep.coll.num_nodes))

# %% [markdown]
# Three levels, although `num_nodes` has only two entries: the longest list decides how many levels there are, and
# levels beyond the end of a shorter list get its last entry. That is why levels 1 and 2 both have 3 nodes.
#
# :::{admonition} Important things to note
# - Not all lists need the same length: the longest defines the number of levels, shorter ones are extended with
#   their last entry.
# - Like most other parameters, `space_transfer_class` and `space_transfer_params` are part of the description.
# - For advanced users: `base_transfer_params` passes parameters to the class that ties two levels together
#   (`BaseTransfer`, which uses the space transfer class), and `base_transfer_class` replaces that class.
# :::
#
# The checks the tests run:

# %%
for l in range(len(S.levels)):
    L = S.levels[l]
    assert (
        L.prob.nvars[0] == problem_params['nvars'][min(l, len(problem_params['nvars']) - 1)]
    ), f"ERROR: number of DOFs is not correct on this level, got {L.prob.nvars}"
    assert (
        L.sweep.coll.num_nodes == sweeper_params['num_nodes'][min(l, len(sweeper_params['num_nodes']) - 1)]
    ), f"ERROR: number of nodes is not correct on this level, got {L.sweep.coll.num_nodes}"
