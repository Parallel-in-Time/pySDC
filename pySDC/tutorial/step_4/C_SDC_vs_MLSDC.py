# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown]
# # Part C: SDC vs. MLSDC
#
# Now we run multilevel SDC and compare it with SDC, for one step of the unforced heat equation. Two descriptions,
# two controllers: SDC on one level, MLSDC on three, coarsened in space and in the number of collocation nodes.

# %%
import matplotlib.pyplot as plt

from pySDC.helpers.stats_helper import get_sorted
from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.implementations.problem_classes.HeatEquation_ND_FD import heatNd_unforced
from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit
from pySDC.implementations.transfer_classes.TransferMesh import mesh_to_mesh

# initialize level parameters
level_params = {'restol': 1e-09, 'dt': 0.1}

# initialize sweeper parameters
sweeper_params_sdc = {'node_type': 'LEGENDRE', 'quad_type': 'RADAU-RIGHT', 'num_nodes': 5, 'QI': 'LU'}
sweeper_params_mlsdc = {'node_type': 'LEGENDRE', 'quad_type': 'RADAU-RIGHT', 'num_nodes': [5, 3, 2], 'QI': 'LU'}

# initialize problem parameters
problem_params_sdc = {
    'nu': 0.1,  # diffusion coefficient
    'freq': 4,  # frequency for the test value
    'nvars': 1023,  # number of degrees of freedom
    'bc': 'dirichlet-zero',  # boundary conditions
}
problem_params_mlsdc = {
    'nu': 0.1,  # diffusion coefficient
    'freq': 4,  # frequency for the test value
    'nvars': [1023, 511, 255],  # number of degrees of freedom for each level
    'bc': 'dirichlet-zero',  # boundary conditions
}

# initialize step parameters
step_params = {'maxiter': 20}

# initialize space transfer parameters
space_transfer_params = {'rorder': 2, 'iorder': 6}

# initialize controller parameters
controller_params = {'logger_level': 30}

# fill description dictionary for SDC
description_sdc = {
    'problem_class': heatNd_unforced,
    'problem_params': problem_params_sdc,
    'sweeper_class': generic_implicit,
    'sweeper_params': sweeper_params_sdc,
    'level_params': level_params,
    'step_params': step_params,
}

# fill description dictionary for MLSDC
description_mlsdc = {
    'problem_class': heatNd_unforced,
    'problem_params': problem_params_mlsdc,
    'sweeper_class': generic_implicit,
    'sweeper_params': sweeper_params_mlsdc,
    'level_params': level_params,
    'step_params': step_params,
    'space_transfer_class': mesh_to_mesh,
    'space_transfer_params': space_transfer_params,
}

# instantiate the controllers
controller_sdc = controller_nonMPI(num_procs=1, controller_params=controller_params, description=description_sdc)
controller_mlsdc = controller_nonMPI(num_procs=1, controller_params=controller_params, description=description_mlsdc)

# %% [markdown]
# To see what the two cost, we also count how often each level evaluates the right-hand side, by wrapping the
# `eval_f` of each level's problem in a counter:


# %%
def count_rhs_evaluations(controller):
    """Wrap eval_f of each level's problem, and return one counter per level"""
    counters = []
    for L in controller.MS[0].levels:
        counter, eval_f = [0], L.prob.eval_f

        def counting_eval_f(*args, eval_f=eval_f, counter=counter, **kwargs):
            counter[0] += 1
            return eval_f(*args, **kwargs)

        L.prob.eval_f = counting_eval_f
        counters.append(counter)
    return counters


rhs_sdc = count_rhs_evaluations(controller_sdc)
rhs_mlsdc = count_rhs_evaluations(controller_mlsdc)

# %% [markdown]
# ## Running both

# %%
# set time parameters
t0 = 0.0
Tend = 0.1

# get initial values on finest level
P = controller_sdc.MS[0].levels[0].prob
uinit = P.u_exact(t0)

# call main functions to get things done...
uend_sdc, stats_sdc = controller_sdc.run(u0=uinit, t0=t0, Tend=Tend)
uend_mlsdc, stats_mlsdc = controller_mlsdc.run(u0=uinit, t0=t0, Tend=Tend)

# get number of iterations for both
niter_sdc = get_sorted(stats_sdc, type='niter', sortby='time')[0][1]
niter_mlsdc = get_sorted(stats_mlsdc, type='niter', sortby='time')[0][1]

# compute exact solution and compare both
uex = P.u_exact(Tend)
err_sdc = abs(uex - uend_sdc)
err_mlsdc = abs(uex - uend_mlsdc)
diff = abs(uend_mlsdc - uend_sdc)

print('Error SDC and MLSDC: %12.8e -- %12.8e' % (err_sdc, err_mlsdc))
print('Difference SDC vs. MLSDC: %12.8e' % diff)
print('Number of iterations SDC and MLSDC: %2i -- %2i' % (niter_sdc, niter_mlsdc))
print('Right-hand side evaluations per level, SDC and MLSDC:', [c[0] for c in rhs_sdc], [c[0] for c in rhs_mlsdc])

# %% tags=["hide-input"]
fig, ax = plt.subplots(figsize=(6, 3.5))
for name, stats in [('SDC', stats_sdc), ('MLSDC', stats_mlsdc)]:
    residual = get_sorted(stats, type='residual_post_iteration', sortby='iter')
    ax.semilogy(*zip(*residual, strict=True), 'o-', label=name)
ax.axhline(level_params['restol'], color='k', ls='--', label='restol')
ax.set_xlabel('iteration')
ax.set_ylabel('residual')
ax.legend(frameon=False)
ax.grid(alpha=0.3)
fig.tight_layout()

# %% [markdown]
# The same result, in half the iterations. This is the best case, and in many situations it cannot be achieved;
# in particular, the interpolation order is crucial.
#
# The count tells the other half of the story: MLSDC evaluates the right-hand side on the finest level exactly as
# often as SDC, plus the evaluations on the coarse levels. After each coarse-level correction, the interpolated
# values need their right-hand side on the fine level again, before the next fine sweep. What halves with the
# iterations are the fine sweeps, and with them the implicit solves on the finest level; the fine right-hand side
# work does not drop at all.
#
# :::{admonition} Try it yourself
# :class: tip
# 1. Interpolate the right-hand side too, instead of evaluating it again: add
#    `description_mlsdc['base_transfer_params'] = {'finter': True}` before the controllers are created, and run
#    everything again.
# 2. Switch on a predictor with the controller parameter `predict_type`: the default `None` has none,
#    `'pfasst_burnin'` sweeps on the coarsest level first. How many iterations does MLSDC need then? (The SDC
#    controller warns that it ignores the predictor: it has only one level.)
# :::
#
# :::{dropdown} Answers
# 1. The iterations and the result stay the same, but the finest level needs 36 evaluations instead of 66.
# 2. 4 instead of 6. The result stays within the tolerance of SDC's: the difference grows from $8 \cdot 10^{-11}$
#    to $2 \cdot 10^{-10}$. The predictor's own sweeps do not count as iterations, though.
# :::
#
# The checks the tests run:

# %%
assert diff < 6e-10, f"ERROR: difference between MLSDC and SDC is higher than expected, got {diff}"
assert niter_sdc - niter_mlsdc >= 6, f"ERROR: MLSDC required more iterations than expected, got {niter_mlsdc}"
