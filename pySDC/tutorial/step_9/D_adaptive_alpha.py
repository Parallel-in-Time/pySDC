# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown]
# # Part D: Adaptive alpha
#
# Parts A to C all picked $\alpha$ by hand and kept it fixed, which means committing to one compromise for the whole
# run. A small $\alpha$ approximates the original problem better and converges in fewer iterations, but conditions
# the diagonalization worse, so round-off and inexact inner solves get amplified. The right balance shifts as the
# residual falls, so a fixed value is wrong at one end of the run or the other.
#
# The `AdaptiveAlpha` convergence controller updates $\alpha$ after every iteration instead, following
# [Čaklović et al.](https://doi.org/10.2140/camcos.2023.18.55):
#
# $$
# \gamma = L (3 \epsilon + \tau), \quad
# \alpha_{k} = \sqrt{\frac{\gamma r_k}{e_k}}, \quad
# e_{k+1} = 2 \sqrt{\gamma e_k r_k},
# $$
#
# with $L$ the number of steps in the block, $\epsilon$ machine precision, $\tau$ the inner solver tolerance, $r_k$
# the residual and $e_k$ a running bound on the error.
#
# ## The setup
#
# We compare a few fixed values of $\alpha$ with the adaptive one on the advection problem of
# [Part C](C_paradiag_in_pySDC), but with a direct solver for the complex shifted systems ParaDiag produces instead
# of GMRES, which struggles with them. The setup lives in `paradiag_setup.py` next to this tutorial, because
# [Part E](E_paradiag_MPI) uses exactly the same. Switching to the adaptive strategy is one entry in the description:
#
# :::{literalinclude} paradiag_setup.py
# :pyobject: get_controller_params
# :::
#
# And `run` sets up the controller, the virtually parallel one when it gets no communicator:
#
# :::{literalinclude} paradiag_setup.py
# :pyobject: run
# :::

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pySDC.tutorial.step_9.paradiag_setup import alpha_settings, format_result, num_steps_total, run

# %% [markdown]
# ## The comparison
#
# One block holding every step, with the virtually parallel controller:

# %%
# one block holding every step; Part E runs the same settings across MPI ranks
block_size = num_steps_total

results = {}
lines = []
for alpha in alpha_settings:
    uend, niter, error, final_alpha = run(alpha, block_size)
    results[alpha] = (uend, niter, final_alpha)
    lines.append(format_result('virtual', alpha, niter, error, final_alpha))

# Part E compares with these results, so they go into a file, too
Path("data").mkdir(parents=True, exist_ok=True)
with open('data/step_9_D_out.txt', 'w') as f:
    for line in lines:
        f.write(line + '\n')
        print(line)

# %% tags=["hide-input"]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3))
labels = [str(alpha) for alpha in alpha_settings]
ax1.bar(labels, [results[alpha][1] for alpha in alpha_settings], color='#4f6bed')
ax1.set_title('iterations', fontsize=10)
ax2.bar(labels, [results[alpha][2] for alpha in alpha_settings], color='#e8743b')
ax2.set_yscale('log')
ax2.set_title(r'$\alpha$ at the end of the run', fontsize=10)
for ax in (ax1, ax2):
    ax.set_xlabel(r'$\alpha$ setting')
fig.tight_layout()

# %% [markdown]
# :::{admonition} Important things to note
# - $\gamma$ is an accuracy floor. There is no point pushing $\alpha$ below the level at which round-off and the
#   inner solver dominate anyway, which is why `inner_tol` enters: a looser inner solve should get a larger $\alpha$.
# - The interesting result is not that the adaptive strategy wins on iteration count. It ties with the best fixed
#   value we tried, but it gets there without being told, and it keeps $\alpha$ orders of magnitude larger while doing
#   so, which is exactly the margin that protects you once the inner solves are inexact.
# - The residual is taken over the whole block, so every rank computes the same $\alpha$ and the controllers stay
#   in step.
# - $\alpha$ is a property of the method, not of the parallelization, so everything here runs with the virtually
#   parallel controller. [Part E](E_paradiag_MPI) takes exactly these settings across MPI ranks and checks they come
#   out the same.
# :::
#
# The checks the tests run:

# %%
# the adaptive strategy should need no more iterations than the best fixed alpha we tried
best_fixed = min(results[a][1] for a in alpha_settings if a != 'adaptive')
assert (
    results['adaptive'][1] <= best_fixed
), f"ERROR: adaptive alpha needed {results['adaptive'][1]} iterations, the best fixed alpha only {best_fixed}"

# alpha changes the iteration, not the problem, so all settings solve the same thing
reference = results[alpha_settings[0]][0]
for alpha in alpha_settings[1:]:
    assert np.allclose(results[alpha][0], reference, atol=1e-5), f'ERROR: alpha {alpha} gives a different solution'
