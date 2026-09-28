# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown]
# # Part C: Studying collocation node types
#
# In [Part B](B_adding_statistics), the energy of the particle changed quite a bit in a single step. Here we test
# whether the collocation nodes are to blame, and on the way show how to set up a parameter study with pySDC:
# describe the whole setup except the parameter to vary, then loop over its values.

# %%
import matplotlib.pyplot as plt
import numpy as np

from pySDC.helpers.stats_helper import get_sorted
from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.implementations.problem_classes.PenningTrap_3D import penningtrap
from pySDC.implementations.sweeper_classes.boris_2nd_order import boris_2nd_order
from pySDC.tutorial.step_3.HookClass_Particles import particle_hook

# initialize level parameters
level_params = {'restol': 1e-06, 'dt': 1.0 / 16}

# initialize sweeper parameters, the node type is set in the loop below
sweeper_params = {'num_nodes': 3}

# initialize problem parameters
problem_params = {
    'omega_E': 4.9,
    'omega_B': 25.0,
    'u0': np.array([[10, 0, 0], [100, 0, 100], [1], [1]], dtype=object),
    'nparts': 1,
    'sig': 0.1,
}

# initialize step parameters
step_params = {'maxiter': 20}

# initialize controller parameters
controller_params = {
    'hook_class': particle_hook,  # specialized hook class for more statistics and output
    'logger_level': 30,  # reduce verbosity of each run
}

# Fill description dictionary for easy hierarchy creation
description = {
    'problem_class': penningtrap,
    'problem_params': problem_params,
    'sweeper_class': boris_2nd_order,
    'level_params': level_params,
    'step_params': step_params,
}

# %% [markdown]
# ## The study
#
# For each node type we build a new controller. That is slightly inefficient, but it makes sure that all variables
# and statistics start afresh. The `stats` of each run go into one dictionary, keyed by the node type.

# %%
# assemble and loop over list of collocation classes
quad_types = ['RADAU-RIGHT', 'GAUSS', 'LOBATTO']
stats_dict = {}
for qtype in quad_types:
    sweeper_params['quad_type'] = qtype
    description['sweeper_params'] = sweeper_params

    # instantiate the controller
    controller = controller_nonMPI(num_procs=1, controller_params=controller_params, description=description)

    # set time parameters
    t0 = 0.0
    Tend = level_params['dt']

    # get initial values on finest level
    P = controller.MS[0].levels[0].prob
    uinit = P.u_init()

    # call main function to get things done...
    uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)

    # gather stats in dictionary, collocation classes being the keys
    stats_dict[qtype] = stats

# %% [markdown]
# Then we compare the energy before and after the step, for each node type:

# %%
ediff = {}
for cclass, stats in stats_dict.items():
    # filter and convert/sort statistics by etot and iterations
    energy = get_sorted(stats, type='etot', sortby='iter')
    # compare base and final energy
    base_energy = energy[0][1]
    final_energy = energy[-1][1]
    ediff[cclass] = abs(base_energy - final_energy)
    print("Energy deviation for %s: %12.8e" % (cclass, ediff[cclass]))

# %% tags=["hide-input"]
fig, ax = plt.subplots(figsize=(5, 3))
ax.bar(list(ediff), list(ediff.values()), color=['#e8743b', '#4f6bed', '#19a979'])
ax.set_yscale('log')
ax.set_ylabel('energy deviation after one step')
ax.axhline(level_params['restol'], color='k', ls='--', label='restol')
ax.legend(frameon=False)
fig.tight_layout()

# %% [markdown]
# Gauss-Radau loses energy, Gauss-Legendre and Gauss-Lobatto hardly any, about $10^{-5}$. The difference is
# symmetry. Both Gauss-Legendre and Gauss-Lobatto nodes are symmetric within the step, Gauss-Radau
# nodes are not.
#
# :::{admonition} Try it yourself
# :class: tip
# Lower the residual tolerance, e.g. to `level_params['restol'] = 1e-10`, and run the study again. What happens to
# the energy deviation of each node type?
# :::
#
# :::{dropdown} Answer
# For the symmetric nodes it drops with the tolerance, to about $6 \cdot 10^{-11}$ (Gauss) and
# $8 \cdot 10^{-10}$ (Lobatto), at the price of a few more iterations. Rule of thumb: energy is conserved to within
# an order of magnitude or so of the residual tolerance. Gauss-Radau stays at 14.5 whatever the tolerance: its error is the one of the
# collocation method itself, not of the iteration. The checks below still pass.
# :::
#
# ## Summary
#
# - A parameter study: set up everything but the parameter, then loop, with a new controller for each value.
# - Working with several `stats` dictionaries is not straightforward, but a meta-dictionary like `stats_dict`
#   helps. Alternatively, process each `stats` right after its run and keep only what you need.
# - Symmetric collocation nodes conserve the energy of this problem to within an order of magnitude or so of the
#   residual tolerance.
#
# The checks the tests run:

# %%
# set expected differences and check
ediff_expect = {'RADAU-RIGHT': 15, 'LOBATTO': 1e-05, 'GAUSS': 3e-05}
for k, v in ediff.items():
    assert v < ediff_expect[k], f"ERROR: energy deviated too much, got {ediff[k]}"
