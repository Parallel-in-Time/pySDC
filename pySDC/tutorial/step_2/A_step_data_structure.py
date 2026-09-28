# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown]
# # Part A: Step data structure
#
# Everything in pySDC is built from **steps**. A step represents one time step with whatever hierarchy we
# prescribe: it holds one or more **levels** (several of them later, for multilevel SDC), and each level holds the
# problem class, the sweeper, the values at the collocation nodes and its parameters.
#
# :::{note}
# This part is for demonstration only: users normally never touch these data structures themselves. The controller
# in [Part C](C_using_pySDCs_frontend) does it for them.
# :::
#
# ## What SDC iterates
#
# [Step 1](../step_1) ended with the collocation problem of one time step, solved directly. For a general
# right-hand side $f$, it reads
#
# $$
# \vec u = \vec u_0 + \Delta t\, Q\, \vec f(\vec u) ,
# $$
#
# with $\vec u$ the values at the $M$ nodes. SDC solves it iteratively: it replaces the full matrix $Q$, which
# couples all nodes, by a lower-triangular approximation $Q_\Delta$, and corrects for the difference with the last
# iterate,
#
# $$
# \vec u^{k+1} - \Delta t\, Q_\Delta\, \vec f(\vec u^{k+1}) = \vec u_0 + \Delta t\, (Q - Q_\Delta)\, \vec f(\vec u^{k}) .
# $$
#
# Because $Q_\Delta$ is lower triangular, this is solved node by node, from the first to the last: one **sweep**, in
# which each node needs one implicit solve the size of the spatial problem, like a step of implicit Euler. Once the
# iteration has converged, $\vec u^{k+1} = \vec u^k$, the $Q_\Delta$ terms cancel, and the result solves the
# collocation problem, whatever $Q_\Delta$ is. $Q_\Delta$ is the **preconditioner** of the iteration: it decides how
# fast SDC converges, not what it converges to. The sweeper parameter `QI` chooses it: `'IE'` is implicit Euler
# from node to node, `'LU'`, used below, takes it from the LU decomposition of $Q^T$ (the "LU trick"). How far an
# iterate is from the solution is measured by the **residual**
# $\|\vec u_0 + \Delta t\, Q\, \vec f(\vec u^{k}) - \vec u^{k}\|$. [Part B](B_my_first_sweeper) runs this iteration by hand.
#
# ## The description
#
# A step is created from a single dictionary, the **description**. It names the classes to use and gives each of
# them its parameters, again as dictionaries.

# %%
from pySDC.core.step import Step
from pySDC.implementations.problem_classes.HeatEquation_ND_FD import heatNd_unforced
from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit

# initialize level parameters
level_params = {
    'restol': 1e-10,  # stop iterating when the residual is below this
    'dt': 0.1,  # time-step size
}

# initialize sweeper parameters
sweeper_params = {
    'quad_type': 'RADAU-RIGHT',  # collocation node type
    'num_nodes': 3,  # number of collocation nodes
    'QI': 'LU',  # preconditioner of the implicit sweep
}

# initialize problem parameters
problem_params = {
    'nu': 0.1,  # diffusion coefficient
    'freq': 4,  # frequency for the test value
    'nvars': 1023,  # number of degrees of freedom
    'bc': 'dirichlet-zero',  # boundary conditions
}

# initialize step parameters
step_params = {
    'maxiter': 20,  # maximum number of iterations
}

# fill description dictionary for easy step instantiation
description = {
    'problem_class': heatNd_unforced,
    'problem_params': problem_params,
    'sweeper_class': generic_implicit,
    'sweeper_params': sweeper_params,
    'level_params': level_params,
    'step_params': step_params,
}

# %% [markdown]
# Note that the description contains *classes*, not instances: `heatNd_unforced`, not `heatNd_unforced(...)`. pySDC
# instantiates them itself, with the parameters from the description. Passing classes around makes life much easier,
# even if it is not the textbook way to program.
#
# ## Creating a step
#
# That is all a step needs:

# %%
# now the description contains more or less everything we need to create a step
S = Step(description=description)

# we only have a single level, make a shortcut
L = S.levels[0]

# one of the integral parts of each level is the problem class, make a shortcut
P = L.prob

print(f'Step with {len(S.levels)} level(s), at most {S.params.maxiter} iterations')
print(f' └─ level {L.level_index}: dt = {L.params.dt}, restol = {L.params.restol}')
print(f'     ├─ problem: {type(P).__name__} with {P.nvars[0]} unknowns')
print(f'     ├─ sweeper: {type(L.sweep).__name__}, {L.sweep.coll.num_nodes} {L.sweep.params.quad_type} nodes')
print(f'     └─ u, f: values at the initial time and the {len(L.u) - 1} nodes, still empty: {L.u}')

# %% [markdown]
# The level holds a list `u` with one entry per collocation node, plus one at the front for the initial value, and
# the same for the right-hand side `f`. They are empty until someone starts the iteration, which we do in
# [Part B](B_my_first_sweeper).
#
# ## The problem inside the step
#
# The problem class inside the level is an ordinary instance, created from `problem_params`. We can run the same
# operator test on it as in [Step 1, Part A](../step_1/A_spatial_problem_setup):

# %%
import numpy as np

# fill u with a sine wave, and compare eval_f with the exact Laplacian
x = np.array([(i + 1) * P.dx for i in range(P.nvars[0])])
u = P.dtype_u(init=P.init)
u[:] = np.sin(np.pi * P.freq[0] * x)
err = abs(P.eval_f(u, 0) + (np.pi * P.freq[0]) ** 2 * P.nu * u)
print(f'Error of the spatial accuracy test: {err:8.6e}')

# %% [markdown]
# Same problem, same parameters, same error: the step built exactly what we built by hand before.
#
# ## Summary
#
# - The `description` dictionary is the central steering tool of pySDC.
# - A step holds levels; a level holds the problem, the sweeper, the values at the nodes, parameters and status.
# - Classes, not instances, go into the description.
#
# The check the tests run:

# %%
assert err <= 2e-04, f"ERROR: the spatial accuracy is higher than expected, got {err}"
