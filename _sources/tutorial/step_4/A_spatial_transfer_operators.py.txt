# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown]
# # Part A: Spatial transfer operators
#
# Multilevel SDC works on a hierarchy of levels, for example several spatial resolutions of the same problem, and
# moves values between them: **restriction** from fine to coarse, **interpolation** (or prolongation) from coarse to
# fine. How this is done depends on the data type, so the user provides it as a `space_transfer_class`. For meshes
# of finite differences, pySDC has `mesh_to_mesh`, which interpolates with Lagrange polynomials.
#
# :::{note}
# As for the sweeper and everything else, users do not instantiate the transfer class themselves when they use a
# controller. We do it here to test it.
# :::

# %%
from collections import namedtuple

import matplotlib.pyplot as plt
import numpy as np

from pySDC.implementations.problem_classes.HeatEquation_ND_FD import heatNd_unforced
from pySDC.implementations.transfer_classes.TransferMesh import mesh_to_mesh

# initialize problem parameters
problem_params = {
    'nu': 0.1,  # diffusion coefficient
    'freq': 3,  # frequency for the test value
    'bc': 'dirichlet-zero',  # boundary conditions
}

# initialize transfer parameters
space_transfer_params = {
    'rorder': 2,  # order of the restriction
    'iorder': 4,  # order of the interpolation
}

# %% [markdown]
# ## One interpolation
#
# A fine problem and a coarse one with half the unknowns (`nvars` of the form $2^p - 1$ halve nicely, as the
# boundary points are not unknowns). `prolong` takes a coarse value to the fine mesh:

# %%
problem_params['nvars'] = 15
Pfine = heatNd_unforced(**problem_params)
problem_params['nvars'] = 7
Pcoarse = heatNd_unforced(**problem_params)
T = mesh_to_mesh(fine_prob=Pfine, coarse_prob=Pcoarse, params=space_transfer_params)

xc = np.array([(i + 1) * Pcoarse.dx for i in range(Pcoarse.nvars[0])])
xf = np.array([(i + 1) * Pfine.dx for i in range(Pfine.nvars[0])])
uc = Pcoarse.dtype_u(Pcoarse.init)
uc[:] = np.sin(np.pi * Pcoarse.freq[0] * xc)
uf = T.prolong(uc)

fig, ax = plt.subplots(figsize=(6, 3))
x = np.linspace(0, 1, 200)
ax.plot(x, np.sin(np.pi * problem_params['freq'] * x), color='0.7', label='exact')
ax.plot(xc, uc, 'o', ms=8, label='coarse, 7 unknowns')
ax.plot(xf, uf, 'x', ms=8, label='interpolated, 15 unknowns')
ax.set_xlabel('$x$')
ax.legend(frameon=False, fontsize=8)
fig.tight_layout()

# %% [markdown]
# ## The order of interpolation
#
# As before, we check the order: interpolate the exact coarse values for a sequence of meshes and compare with the
# exact fine values. We expect 4, the `iorder` we asked for.

# %%
nvars_fine_list = [2**p - 1 for p in range(5, 10)]

# setup id for gathering the results (will sort by nvars)
ID = namedtuple('ID', 'nvars_fine')

# set up dictionary to store results (plus lists)
results = {'nvars_list': nvars_fine_list}

for nvars_fine in nvars_fine_list:
    print('Working on nvars_fine = %4i...' % nvars_fine)

    # instantiate fine problem
    problem_params['nvars'] = nvars_fine  # number of degrees of freedom
    Pfine = heatNd_unforced(**problem_params)

    # instantiate coarse problem using half of the DOFs
    problem_params['nvars'] = int((nvars_fine + 1) / 2.0 - 1)
    Pcoarse = heatNd_unforced(**problem_params)

    # instantiate spatial interpolation
    T = mesh_to_mesh(fine_prob=Pfine, coarse_prob=Pcoarse, params=space_transfer_params)

    # set exact fine solution to compare with
    xvalues_fine = np.array([(i + 1) * Pfine.dx for i in range(Pfine.nvars[0])])
    uexact_fine = Pfine.dtype_u(Pfine.init)
    uexact_fine[:] = np.sin(np.pi * Pfine.freq[0] * xvalues_fine)

    # set exact coarse solution as source
    xvalues_coarse = np.array([(i + 1) * Pcoarse.dx for i in range(Pcoarse.nvars[0])])
    uexact_coarse = Pfine.dtype_u(Pcoarse.init)
    uexact_coarse[:] = np.sin(np.pi * Pcoarse.freq[0] * xvalues_coarse)

    # do the interpolation/prolongation
    uinter = T.prolong(uexact_coarse)

    # compute error and store
    id = ID(nvars_fine=nvars_fine)
    results[id] = abs(uinter - uexact_fine)

# %%
# observed order between consecutive meshes, as in step 1, part B
nvars = sorted(results['nvars_list'])
errors = [results[ID(nvars_fine=n)] for n in nvars]
orders = [np.log(errors[i - 1] / errors[i]) / np.log(nvars[i] / nvars[i - 1]) for i in range(1, len(nvars))]

for p in range(len(orders)):
    print(
        'Expected order %2i, got order %5.2f, deviation of %5.2f%%'
        % (
            space_transfer_params['iorder'],
            orders[p],
            100 * abs(space_transfer_params['iorder'] - orders[p]) / space_transfer_params['iorder'],
        )
    )

# %% [markdown]
# :::{admonition} Important things to note
# - MLSDC and PFASST rely on high-order interpolation in space. With Lagrange-based interpolation, orders of 4 to 6
#   and above are recommended.
# - Restriction, however, can be of order 2, and is thus not tested here.
# :::
#
# :::{admonition} Try it yourself
# :class: tip
# Set `space_transfer_params['iorder'] = 6` and run the order check again.
# :::
#
# :::{dropdown} Answer
# Sixth order, but approached from above: 6.45, 6.83, 6.39 and finally 5.98 on the finest meshes. The coarse ones
# are not in the asymptotic regime yet, so the check below, which allows 5% everywhere, fails.
# :::
#
# The check the tests run:

# %%
for order in orders:
    assert (
        abs(space_transfer_params['iorder'] - order) / space_transfer_params['iorder'] < 0.05
    ), f'ERROR: did not get expected orders for interpolation, got {order}'
