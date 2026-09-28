# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown]
# # Part B: My first sweeper
#
# Now we run SDC for one time step, by hand. The problem is the heat equation with a forcing term,
# `heatNd_forced`, and the sweeper `imex_1st_order` treats the two parts differently: diffusion implicitly,
# forcing explicitly. This is **IMEX SDC**, and it needs the right-hand side in two parts, which the data type
# `imex_mesh` provides as `.impl` and `.expl`.
#
# :::{note}
# Again, this is for demonstration only. Users normally let a controller do all of this, see
# [Part C](C_using_pySDCs_frontend).
# :::

# %%
import matplotlib.pyplot as plt

from pySDC.core.step import Step
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

# Fill description dictionary for easy hierarchy creation
description = {
    'problem_class': heatNd_forced,
    'problem_params': problem_params,
    'sweeper_class': imex_1st_order,
    'sweeper_params': sweeper_params,
    'level_params': level_params,
    'step_params': step_params,
}

# instantiate the step we are going to work on, and make shortcuts for the level and the problem
S = Step(description=description)
L = S.levels[0]
P = L.prob

# %% [markdown]
# ## Parameters and status
#
# Steps and levels carry two kinds of attributes. **Parameters** are what we asked for in the description, such as
# `S.params.maxiter` or `L.params.restol`. The **status** reflects where the computation currently is, such as
# `S.status.iter`, `L.status.time` or `L.status.residual`. We set the status of the level to the start of the step
# and put the initial value at index 0:

# %%
# set initial time in the status of the level
L.status.time = 0.1
# compute initial value (using the exact function here)
L.u[0] = P.u_exact(L.time)

# %% [markdown]
# ## Predict, sweep, check
#
# The sweeper does the actual work, with three routines. `predict` fills the nodes with a first guess; without it,
# they stay empty. `compute_residual` measures how far the values at the nodes are from solving the collocation
# problem. `update_nodes` is one SDC sweep.

# %%
# access the sweeper's predict routine to get things started
L.sweep.predict()
# compute the residual (we may be done already!)
L.sweep.compute_residual()

print(f'right-hand side at node 1: {type(L.f[1]).__name__} with parts .impl and .expl')
print(f'residual of the initial guess: {L.status.residual:12.8e}')

# %% [markdown]
# Now we sweep until the residual is below `restol` or we run out of iterations:

# %%
# reset iteration counter
S.status.iter = 0
residuals = [L.status.residual]
# run the SDC iteration until either the maximum number of iterations is reached or the residual is small enough
while S.status.iter < S.params.maxiter and L.status.residual > L.params.restol:
    # this is where the nodes are actually updated according to the SDC formulas
    L.sweep.update_nodes()
    # compute/update the residual
    L.sweep.compute_residual()
    # increment the iteration counter
    S.status.iter += 1
    residuals.append(L.status.residual)
    print(
        f'Time {L.time:4.2f} of {L.level_index} -- Iteration: {S.status.iter:2d} -- Residual: {L.status.residual:12.8e}'
    )

# %% tags=["hide-input"]
fig, ax = plt.subplots(figsize=(6, 3.5))
ax.semilogy(residuals, 'o-')
ax.axhline(L.params.restol, color='k', ls='--', label='restol')
ax.set_xlabel('iteration')
ax.set_ylabel('residual')
ax.grid(alpha=0.3)
ax.legend(frameon=False)
fig.tight_layout()

# %% [markdown]
# Most sweeps reduce the residual by a factor of five to ten, with a slower patch around the sixth. Once converged,
# the values at the nodes solve a collocation problem like the one in
# [Step 1, Part C](../step_1/C_collocation_problem_setup), without ever assembling it.
#
# ## Finishing the step
#
# The solution at the end of the step is computed from the values at the nodes by `compute_end_point`, and only
# there. Then the time moves on and we compare with the exact solution:

# %%
# compute the interval's endpoint: this (and only this) will set uend, depending on the collocation nodes
L.sweep.compute_end_point()
# update the simulation time
L.status.time += L.dt

# compute exact solution and compare
err = abs(P.u_exact(L.status.time) - L.uend)
res, niter = L.status.residual, S.status.iter
print(f'Error and residual: {err:12.8e} -- {res:12.8e}')

# %% [markdown]
# :::{admonition} Try it yourself
# :class: tip
# The sweeper preconditions each sweep with a lower-triangular matrix $Q_\Delta$, chosen with the sweeper parameter
# `QI`. The default is `'IE'` (implicit Euler). Set `sweeper_params['QI'] = 'LU'` in the first cell, or
# `'MIN-SR-S'`, and run everything again. How many iterations do you need now?
# :::
#
# :::{dropdown} Answer
# `'LU'` needs 10 iterations instead of 12, `'MIN-SR-S'` needs 11. The error stays the same: all of them converge
# to the same collocation solution, only the path differs. `'MIN-SR-S'` is diagonal, so its nodes could be solved in
# parallel, which is what the project [parallel SDC](../../projects/parallelSDC) is about.
# :::
#
# ## Summary
#
# - One SDC step is: set the initial value, `predict`, then `update_nodes` and `compute_residual` until converged,
#   then `compute_end_point`.
# - Parameters are what you asked for; status is where the computation is.
# - This logic is simple but tedious, and it gets much worse with several steps, levels or processes. Hence
#   [Part C](C_using_pySDCs_frontend).
#
# The checks the tests run:

# %%
assert err <= 1e-5, f"ERROR: IMEX SDC iteration did not reduce the error enough, got {err}"
assert res <= level_params['restol'], f"ERROR: IMEX SDC iteration did not reduce the residual enough, got {res}"
assert niter <= 12, f"ERROR: IMEX SDC took too many iterations, got {niter}"
