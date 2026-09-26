# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown]
# # Part B: My first PFASST run
#
# With the multistep multilevel hierarchy of [Part A](A_multistep_multilevel_hierarchy), we are ready for PFASST,
# starting with the heat equation. One of the most important characteristics of a parallel-in-time method is how it
# behaves when more time steps are computed in parallel for the same problem, i.e. with `dt` and `Tend` fixed. So we
# run the 16 time steps of $[0, 4]$ with 1, 2, 4, 8 and 16 parallel steps.

# %%
import matplotlib.pyplot as plt
import numpy as np

from pySDC.helpers.stats_helper import get_sorted
from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.implementations.problem_classes.HeatEquation_ND_FD import heatNd_forced
from pySDC.implementations.sweeper_classes.imex_1st_order import imex_1st_order
from pySDC.implementations.transfer_classes.TransferMesh import mesh_to_mesh

# initialize level parameters
level_params = {'restol': 1e-10, 'dt': 0.25}

# initialize sweeper parameters
sweeper_params = {
    'quad_type': 'RADAU-RIGHT',
    'num_nodes': [3],
    'QI': 'LU',  # For the IMEX sweeper, the LU-trick can be activated for the implicit part
}

# initialize problem parameters
problem_params = {
    'nu': 0.1,  # diffusion coefficient
    'freq': 8,  # frequency for the test value
    'nvars': [511, 255],  # number of degrees of freedom for each level
    'bc': 'dirichlet-zero',  # boundary conditions
}

# initialize step parameters
step_params = {'maxiter': 50}

# initialize space transfer parameters
space_transfer_params = {'rorder': 2, 'iorder': 6}

# initialize controller parameters
controller_params = {
    'logger_level': 30,
    'predict_type': 'pfasst_burnin',  # PFASST's coarse-level predictor, see step 4, part C
}

# fill description dictionary for easy step instantiation
description = {
    'problem_class': heatNd_forced,  # pass problem class
    'problem_params': problem_params,  # pass problem parameters
    'sweeper_class': imex_1st_order,  # pass sweeper
    'sweeper_params': sweeper_params,  # pass sweeper parameters
    'level_params': level_params,  # pass level parameters
    'step_params': step_params,  # pass step parameters
    'space_transfer_class': mesh_to_mesh,  # pass spatial transfer class
    'space_transfer_params': space_transfer_params,  # pass parameters for spatial transfer
}

# set time parameters
t0 = 0.0
Tend = 4.0

# set up list of parallel time-steps to run PFASST with
nsteps = int(Tend / level_params['dt'])
num_proc_list = [2**i for i in range(int(np.log2(nsteps) + 1))]
print('numbers of parallel time steps:', num_proc_list)

# %% [markdown]
# ## The runs
#
# One controller for each number of parallel steps; the rest is what we did in [Step 3](../step_3), for each run:
# the error, and statistics of the iteration counts.

# %% tags=["scroll-output"]
results = {}
# loop over different number of processes and check results
for num_proc in num_proc_list:
    print('Working with %2i processes...' % num_proc)
    # instantiate controller
    controller = controller_nonMPI(num_procs=num_proc, controller_params=controller_params, description=description)

    # get initial values on finest level
    P = controller.MS[0].levels[0].prob
    uinit = P.u_exact(t0)

    # call main function to get things done...
    uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)

    # compute exact solution and compare
    uex = P.u_exact(Tend)
    err = abs(uex - uend)

    # filter statistics by type (number of iterations)
    iter_counts = get_sorted(stats, type='niter', sortby='time')

    # compute and print statistics
    for item in iter_counts:
        print('Number of iterations for time %4.2f: %2i' % item)
    print()
    niters = np.array([item[1] for item in iter_counts])
    print('   Mean number of iterations: %4.2f' % np.mean(niters))
    print('   Range of values for number of iterations: %2i ' % np.ptp(niters))
    print('   Position of max/min number of iterations: %2i -- %2i' % (int(np.argmax(niters)), int(np.argmin(niters))))
    print('   Std and var for number of iterations: %4.2f -- %4.2f' % (float(np.std(niters)), float(np.var(niters))))
    print()
    print()

    results[num_proc] = (err, niters)

# %% [markdown]
# The iteration counts of all runs at a glance, one row per number of parallel steps:

# %% tags=["hide-input"]
fig, ax = plt.subplots(figsize=(8, 2.6), constrained_layout=True)
image = ax.imshow([results[n][1] for n in num_proc_list], cmap='viridis', aspect='auto', vmin=0)
ax.set_yticks(range(len(num_proc_list)))
ax.set_yticklabels(num_proc_list)
ax.set_ylabel('parallel steps')
ax.set_xlabel('time step')
fig.colorbar(image, label='iterations')

# %% [markdown]
# PFASST performs very well here: whether the 16 steps are computed one after another or all at once, they need
# about 5 iterations each.
#
# :::{admonition} Important things to note
# - In the IMEX sweeper, `QI = 'LU'` activates the LU-trick for the implicit part. For stiff parabolic problems
#   with Gauss-Radau nodes, this is usually a very good idea.
# - As usual for MLSDC and PFASST, success depends heavily on the parameters. A more complicated problem, a more
#   or less stiff one, a different order of the spatial interpolation, etc. can give completely different results.
#   [Part C](C_advection_and_PFASST) has an example.
# :::
#
# The checks the tests run, for every number of parallel steps:

# %%
for err, niters in results.values():
    assert err < 1.3505e-04, f"ERROR: error is too high, got {err}"
    assert np.ptp(niters) <= 1, f"ERROR: range of number of iterations is too high, got {np.ptp(niters)}"
    assert np.mean(niters) <= 5.0, f"ERROR: mean number of iterations is too high, got {np.mean(niters)}"
