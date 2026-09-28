# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown]
# # Part C: Using pySDC's frontend
#
# Finally, the user-friendliest interface pySDC has to offer: a **controller** does the whole iteration logic of
# [Part B](B_my_first_sweeper) for us. We use `controller_nonMPI`, the default controller. It needs no `mpi4py`, and
# depending on the description it can do SDC, multilevel SDC, multistep SDC and PFASST (more on this in the next
# steps).
#
# The description is the same as in Part B. New are the **controller parameters**, here used to also write the
# log to a file.

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.implementations.problem_classes.HeatEquation_ND_FD import heatNd_forced
from pySDC.implementations.sweeper_classes.imex_1st_order import imex_1st_order

# initialize level parameters
level_params = {'restol': 1e-10, 'dt': 0.1}

# initialize sweeper parameters
sweeper_params = {'quad_type': 'RADAU-RIGHT', 'num_nodes': 3}

# initialize problem parameters
problem_params = {
    'nu': 0.1,  # diffusion coefficient
    'freq': 4,  # frequency for the test value
    'nvars': 1023,  # number of degrees of freedom
    'bc': 'dirichlet-zero',  # boundary conditions
}

# initialize step parameters
step_params = {'maxiter': 20}

# initialize controller parameters
controller_params = {
    'log_to_file': True,
    'fname': 'data/step_2_C_out.txt',
}

# Fill description dictionary for easy hierarchy creation
description = {
    'problem_class': heatNd_forced,
    'problem_params': problem_params,
    'sweeper_class': imex_1st_order,
    'sweeper_params': sweeper_params,
    'level_params': level_params,
    'step_params': step_params,
}

Path("data").mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## Instantiating the controller
#
# When the controller is created, it prints the full setup. Everything marked with `-->` was set by us, `->` marks
# what follows from that, and the rest are defaults. This is the place to check what a run actually used. Set the
# controller parameter `dump_setup` to `False` to switch it off.

# %% tags=["scroll-output"]
# instantiate the controller
controller = controller_nonMPI(num_procs=1, controller_params=controller_params, description=description)

# %% [markdown]
# ## Running
#
# We ask for the time interval $[0.1, 0.3]$, which with `dt = 0.1` means two time steps. The initial value comes
# from the problem class, which the controller has created inside its step, just like in Part A.

# %%
# set time parameters
t0 = 0.1
Tend = 0.3  # note that we are requesting 2 time steps here (dt is 0.1)

# get initial values on finest level
P = controller.MS[0].levels[0].prob
uinit = P.u_exact(t0)

# call main function to get things done...
uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)

# compute exact solution and compare
uex = P.u_exact(Tend)
err = abs(uex - uend)
print(f'Error after SDC iterations: {err:8.6e}')

# %% [markdown]
# That is the whole program a user writes: a description, some pre- and some post-processing. The controller also
# wrote its log to the file we named; here are its last lines:

# %%
print(''.join(Path(controller_params['fname']).read_text().splitlines(True)[-3:]))

# %% tags=["hide-input"]
x = np.array([(i + 1) * P.dx for i in range(P.nvars[0])])
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3))
ax1.plot(x, uinit, label=f'$t = {t0}$')
ax1.plot(x, uend, label=f'$t = {Tend}$, SDC')
ax1.plot(x[::40], uex[::40], 'k.', label=f'$t = {Tend}$, exact')
ax1.set_xlabel('$x$')
ax1.legend(frameon=False, fontsize=8)
ax2.plot(x, np.abs(uend - uex))
ax2.set_xlabel('$x$')
ax2.set_title('error at $t = 0.3$', fontsize=10)
fig.tight_layout()

# %% [markdown]
# :::{admonition} Important things to note
# - The description is all a user needs to steer pySDC; most of the logic and the data structures stay hidden.
#   This part is the prototype for working with pySDC.
# - Unlike in Part B, we have no direct access to the residuals or the iteration counts yet: they are in `stats`,
#   which [Step 3](../step_3) is about.
# :::
#
# The check the tests run:

# %%
assert err <= 2e-5, f"ERROR: controller doing IMEX SDC iteration did not reduce the error enough, got {err}"
