# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown]
# # Part C: Advection and PFASST
#
# PFASST did very well for the heat equation in [Part B](B_my_first_PFASST_run). Now the same test for an
# advection problem, $u_t + c\, u_x = 0$ with periodic boundaries. The setup is the same, but with the fully
# implicit sweeper `generic_implicit`, and with two preconditioners to compare: the LU-trick and implicit Euler.
#
# The setup is periodic in time as well: at `Tend = 1`, the wave has travelled once through the domain, and the
# exact solution looks exactly like the initial condition.

# %%
import matplotlib.pyplot as plt
import numpy as np

from pySDC.helpers.stats_helper import get_sorted
from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.implementations.problem_classes.AdvectionEquation_ND_FD import advectionNd
from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit
from pySDC.implementations.transfer_classes.TransferMesh import mesh_to_mesh

# initialize level parameters
level_params = {'restol': 1e-09, 'dt': 0.0625}

# initialize sweeper parameters, the preconditioner QI is set in the loop below
sweeper_params = {'quad_type': 'RADAU-RIGHT', 'num_nodes': [3]}

# initialize problem parameters
problem_params = {
    'c': 1,  # advection coefficient
    'freq': 4,  # frequency for the test value
    'nvars': [128, 64],  # number of degrees of freedom for each level
    'order': 4,
    'bc': 'periodic',
    'stencil_type': 'center',
}

# initialize step parameters
step_params = {'maxiter': 50}

# initialize space transfer parameters
space_transfer_params = {'rorder': 2, 'iorder': 6, 'periodic': True}

# initialize controller parameters
controller_params = {'logger_level': 30, 'predict_type': 'pfasst_burnin'}

# fill description dictionary for easy step instantiation
description = {
    'problem_class': advectionNd,  # pass problem class
    'problem_params': problem_params,  # pass problem parameters
    'sweeper_class': generic_implicit,  # pass sweeper
    'level_params': level_params,  # pass level parameters
    'step_params': step_params,  # pass step parameters
    'space_transfer_class': mesh_to_mesh,  # pass spatial transfer class
    'space_transfer_params': space_transfer_params,  # pass parameters for spatial transfer
}

# set time parameters
t0 = 0.0
Tend = 1.0

# set up list of parallel time-steps to run PFASST with
nsteps = int(Tend / level_params['dt'])
num_proc_list = [2**i for i in range(int(np.log2(nsteps) + 1))]

# set up list of types of implicit SDC sweepers: LU and implicit Euler here
QI_list = ['LU', 'IE']

# %% [markdown]
# ## The runs
#
# As in Part B, for each preconditioner and each number of parallel steps:

# %% tags=["scroll-output"]
results = {}
# loop over different types of implicit sweeper types
for QI in QI_list:
    # define and set preconditioner for the implicit sweeper
    sweeper_params['QI'] = QI
    description['sweeper_params'] = sweeper_params  # pass sweeper parameters

    # loop over different number of processes
    for num_proc in num_proc_list:
        print('Working with QI = %s on %2i processes...' % (QI, num_proc))
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
        niters = np.array([item[1] for item in iter_counts])
        print('   Mean number of iterations: %4.2f' % np.mean(niters))
        print('   Range of values for number of iterations: %2i ' % np.ptp(niters))
        print(
            '   Position of max/min number of iterations: %2i -- %2i' % (int(np.argmax(niters)), int(np.argmin(niters)))
        )
        print(
            '   Std and var for number of iterations: %4.2f -- %4.2f' % (float(np.std(niters)), float(np.var(niters)))
        )
        print()

        results[QI, num_proc] = (err, niters)

    means = [np.mean(results[QI, n][1]) for n in num_proc_list]
    print('Mean number of iterations went up from %4.2f to %4.2f for QI = %s!' % (min(means), max(means), QI))
    print()
    print()

# %% tags=["hide-input"]
fig, axes = plt.subplots(1, len(QI_list), figsize=(10, 2.6), sharey=True, constrained_layout=True)
vmax = max(max(niters) for _, niters in results.values())
for ax, QI in zip(axes, QI_list, strict=True):
    image = ax.imshow([results[QI, n][1] for n in num_proc_list], cmap='viridis', aspect='auto', vmin=0, vmax=vmax)
    ax.set_title(f'QI = {QI}', fontsize=10)
    ax.set_xlabel('time step')
axes[0].set_yticks(range(len(num_proc_list)))
axes[0].set_yticklabels(num_proc_list)
axes[0].set_ylabel('parallel steps')
fig.colorbar(image, ax=axes, label='iterations')

# %% [markdown]
# Unlike for the heat equation, the iteration counts grow significantly with the number of parallel steps, and
# within each block of parallel steps the later ones need more. How much depends on the problem, but this is typical
# for parallel-in-time methods of this kind. Note also that the LU-trick, so good for the heat equation, is no
# better than implicit Euler here, and even slightly worse with 8 and 16 parallel steps.
#
# :::{admonition} Try it yourself
# :class: tip
# Add `'MIN-SR-S'` to `QI_list`: a diagonal preconditioner, whose nodes could be solved in parallel. How does it
# fare with 16 parallel steps?
# :::
#
# :::{dropdown} Answer
# Much worse: about 24 iterations per step on average, against 15 for `'LU'` and 13 for `'IE'`. With one step at a
# time, all three need 5. With 16 parallel steps, the last four hit `maxiter = 50` without reaching `restol`, so its
# error, $5.2 \cdot 10^{-4}$, is above what the check in the last cell allows. All converged runs end at the same
# error, set by the spatial discretization, just below that threshold.
# :::
#
# :::{admonition} Important things to note
# - Like the IMEX sweeper, `generic_implicit` lets you choose the preconditioner with `QI`: `'IE'` for implicit
#   Euler, `'LU'` for the LU-trick, and more. They come from the [qmat](https://github.com/Parallel-in-Time/qmat)
#   package, and `qmat.qdelta.QDELTA_GENERATORS` lists them all, implicit and explicit ones.
# :::
#
# The check the tests run, for every run:

# %%
for err, _ in results.values():
    assert err < 5.1365e-04, f"ERROR: error is too high, got {err}"
