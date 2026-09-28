# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown]
# # Part B: Adding statistics
#
# Now we extend the statistics with entries of our own. To make things more interesting (and complicated), this
# part also brings a new problem, a new sweeper and new data types:
#
# - `PenningTrap_3D`: particles in a Penning trap, held in place by an electric and a magnetic field,
# - `boris_2nd_order`: SDC for second-order problems with the Boris method, see
#   [this paper](http://dx.doi.org/10.1016/j.jcp.2015.04.022),
# - `particles` for positions, velocities, charges and masses, and `fields` for the electric and magnetic field.
#
# An important measure for this kind of problem is the total energy of the system, which we would like to compute
# after each time step. That is a job for a new **hook**.
#
# ## A hook of our own
#
# Hooks are called by the controller at fixed points of a run: before it, before and after each step, iteration
# or sweep, and after it. `particle_hook`, in `HookClass_Particles.py` next to this tutorial, computes the total
# energy before the run and after each step and adds it to the statistics with `add_to_stats`, as type `'etot'`.
# This is what it does after a step:
#
# :::{literalinclude} HookClass_Particles.py
# :pyobject: particle_hook.post_step
# :::
#
# Our hook does not replace pySDC's default statistics: those come from `DefaultHooks`, which the controller always
# adds next to ours. The call to `super()` first still matters: the base class keeps track of restarted steps, and
# `add_to_stats` writes that into the key of each entry.

# %%
import numpy as np
from pathlib import Path

from pySDC.helpers.stats_helper import get_list_of_types, get_sorted
from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.implementations.problem_classes.PenningTrap_3D import penningtrap
from pySDC.implementations.sweeper_classes.boris_2nd_order import boris_2nd_order
from pySDC.tutorial.step_3.HookClass_Particles import particle_hook

# initialize level parameters
level_params = {'restol': 1e-08, 'dt': 1.0 / 16}

# initialize sweeper parameters
sweeper_params = {'quad_type': 'RADAU-RIGHT', 'num_nodes': 3}

# initialize problem parameters for the Penning trap
problem_params = {
    'omega_E': 4.9,  # E-field frequency
    'omega_B': 25.0,  # B-field frequency
    'u0': np.array([[10, 0, 0], [100, 0, 100], [1], [1]], dtype=object),  # initial position, velocity, charge, mass
    'nparts': 1,  # number of particles in the trap
    'sig': 0.1,  # smoothing parameter for the forces
}

# initialize step parameters
step_params = {'maxiter': 20}

# initialize controller parameters
controller_params = {
    'hook_class': particle_hook,  # specialized hook class for more statistics and output
    'log_to_file': True,
    'fname': 'data/step_3_B_out.txt',
}

# Fill description dictionary for easy hierarchy creation
description = {
    'problem_class': penningtrap,
    'problem_params': problem_params,
    'sweeper_class': boris_2nd_order,
    'sweeper_params': sweeper_params,
    'level_params': level_params,
    'step_params': step_params,
}

Path("data").mkdir(parents=True, exist_ok=True)

# %% [markdown]
# The hook goes into the controller parameters. As in [Step 2](../step_2/C_using_pySDCs_frontend), the controller
# prints its setup when we create it, and the log of the run:

# %% tags=["scroll-output"]
# instantiate the controller
controller = controller_nonMPI(num_procs=1, controller_params=controller_params, description=description)

# set time parameters: a single step
t0 = 0.0
Tend = level_params['dt']

# get initial values on finest level
P = controller.MS[0].levels[0].prob
uinit = P.u_init()

# call main function to get things done...
uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)

# %% [markdown]
# ## Particles
#
# The solution is a `particles` data type. Its parts are separate arrays, one column per particle:

# %%
print('position:', uend.pos.T, '\nvelocity:', uend.vel.T, '\ncharge:', uend.q, ' mass:', uend.m)

# %% [markdown]
# ## Our statistics
#
# Our type `'etot'` now shows up among the others, and `get_sorted` treats it like any other:

# %%
print('etot is registered:', 'etot' in get_list_of_types(stats))

# filter statistics type (etot)
energy = get_sorted(stats, type='etot', sortby='iter')

# get base energy and show difference
base_energy = energy[0][1]
for item in energy:
    print(
        'Total energy and deviation in iteration %2i: %12.10f -- %12.8e'
        % (item[0], item[1], abs(base_energy - item[1]))
    )

# %% [markdown]
# Iteration 0 is the energy the hook computed before the run, the other one after the step, at the iteration it
# converged in. For this single particle the exact solution is known, so we can also check the position:

# %%
# compute error compared to know exact solution for one particle
uex = P.u_exact(Tend)
err = np.linalg.norm(uex.pos - uend.pos, np.inf) / np.linalg.norm(uex.pos, np.inf)
print(f'relative error of the position: {err:.3e}')

# %% [markdown]
# The position is accurate, but the energy has changed by about 14.5 out of 8800 in a single step. We look into
# that in [Part C](C_study_collocations).
#
# :::{admonition} Important things to note
# - A custom hook calls `super()` in every method it overrides, so that its entries are labelled correctly when a
#   step is restarted, e.g. by adaptivity.
# - User-defined statistics can also come from the problem class: give it an attribute (e.g. the number of GMRES
#   iterations of its spatial solver) and read it in the hook through the level, as `L.prob`.
# :::
#
# The checks the tests run:

# %%
assert abs(base_energy - energy[-1][1]) < 15, f'ERROR: energy deviated too much, got {base_energy - energy[-1][1]}'
assert err < 5e-04, f"ERROR: solution is not as exact as expected, got {err}"
