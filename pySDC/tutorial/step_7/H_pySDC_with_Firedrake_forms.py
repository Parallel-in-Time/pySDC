# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#   kernelspec:
#     display_name: Python 3
#     name: python3
#   language_info:
#     name: python
# ---

# %% [markdown]
# # Part H: pySDC from Firedrake forms
#
# In [Part E](E_pySDC_with_Firedrake), the heat equation comes with a hand-written problem class: it sets up its own
# Firedrake solvers for the right-hand side and for the implicit systems pySDC needs. Here we skip that class. We write
# down the weak form of the equation, as we would for Firedrake anyway, and the generic problem class
# [`FormFiredrake`](https://github.com/Parallel-in-Time/pySDC/blob/master/pySDC/implementations/problem_classes/FormFiredrake.py)
# builds everything pySDC needs from it.
#
# ## The equation and its weak form
#
# We solve a reaction-diffusion equation of Allen-Cahn type on the unit square,
#
# $$
# u_t = \nu \Delta u + u - u^3 + g,
# $$
#
# with Dirichlet boundary values. The forcing $g$ and the boundary values come from a solution $u_e(x, y, t)$ we pick,
# so that we can measure the error: this is the *method of manufactured solutions*.
#
# Finite elements turn the equation into a system of ordinary differential equations, $M u' = R(u)$, for the
# coefficients $u$ of the finite element function. The *mass matrix* $M$ and the right-hand side $R$ come from
# multiplying the equation by a test function $v$ and integrating over the domain, integrating the Laplacian by parts:
#
# $$
# \underbrace{\int u_t\, v\, dx}_{m(u_t, v)}
# = \underbrace{-\nu \int \nabla u \cdot \nabla v\, dx}_{\text{diffusion}}
# + \underbrace{\int (u - u^3)\, v\, dx}_{\text{reaction}} + \int g\, v\, dx.
# $$
#
# These *forms* are all `FormFiredrake` needs. We give each as a Python function of the solution `u` and the test
# function `v`, written in [UFL](https://fenics.readthedocs.io/projects/ufl), the form language Firedrake uses. Time
# enters through a Firedrake `Constant` `t`: pySDC sets it to the right time before every evaluation and solve, so
# forms, boundary values and the exact solution may all depend on it.
#
# The forcing is the one part we do not write by hand. `manufactured_forcing` inserts $u_e$ into the forms, and UFL
# differentiates it in time, so $g$ is exact for whatever $u_e$ we choose. Ours is quadratic in space, so the finite
# elements represent it exactly and all of the error we see comes from the time stepping.

# %%
from pathlib import Path

import firedrake as fd

from pySDC.helpers.ufl_forms import manufactured_forcing, total_rhs

NU = 0.1


def get_forms(imex, n=8):
    """
    The problem as forms, for `FormFiredrake` or `FormFiredrakeIMEX`.

    Args:
        imex (bool): Whether to put the reaction into the explicit part
        n (int): Number of elements per direction

    Returns:
        dict: The problem parameters
    """
    mesh = fd.UnitSquareMesh(n, n)
    V = fd.FunctionSpace(mesh, 'CG', 2)
    x, y = fd.SpatialCoordinate(mesh)
    t = fd.Constant(0.0)

    u_exact = fd.cos(t) * (1 + x**2 + 2 * y**2) / 4

    def mass(u, v):
        return u * v * fd.dx

    def diffusion(u, v):
        return -NU * fd.inner(fd.grad(u), fd.grad(v)) * fd.dx

    def reaction(u, v):
        return (u - u**3) * v * fd.dx

    forcing = manufactured_forcing(mass, total_rhs(diffusion, reaction), u_exact, t, fd.TestFunction(V))

    params = {
        'V': V,
        'mass': mass,
        't': t,
        'u_exact': u_exact,
        # a tuple: pySDC would read a list as one entry per level
        'bcs': (fd.DirichletBC(V, u_exact, 'on_boundary'),),
    }
    if imex:
        params['rhs'] = lambda u, v: diffusion(u, v) + forcing
        params['rhs_exp'] = reaction
        # the implicit systems are linear now: one linear solve, no Newton iteration around it
        params['solver_parameters'] = {'snes_type': 'ksponly'}
    else:
        params['rhs'] = lambda u, v: diffusion(u, v) + reaction(u, v) + forcing
    return params


# %% [markdown]
# ## Implicit or IMEX
#
# `FormFiredrake` treats the whole right-hand side implicitly, so every SDC node needs a nonlinear solve. Firedrake
# does it with Newton's method, and the Jacobian comes from differentiating the forms symbolically. `FormFiredrakeIMEX`
# takes a second, explicit right-hand side: here the reaction, so the implicit systems become linear. Either way, the
# problem class never inverts the mass matrix. It hands pySDC $M u$ and $R(u)$ instead, which is what the
# *mass-matrix sweepers* `generic_implicit_mass` and `imex_1st_order_mass` work with.
#
# The boundary values enter only in the solves. As a result, the rows of the SDC residual that belong to boundary
# nodes carry no equation, and the problem class leaves them out of the residual it reports.


# %%
def setup(imex):
    """
    Helper routine to set up parameters

    Args:
        imex (bool): Whether to run IMEX SDC instead of fully implicit SDC

    Returns:
        description and controller_params parameter dictionaries
    """
    from pySDC.implementations.problem_classes.FormFiredrake import FormFiredrake, FormFiredrakeIMEX
    from pySDC.implementations.sweeper_classes.generic_implicit_mass import generic_implicit_mass
    from pySDC.implementations.sweeper_classes.imex_1st_order_mass import imex_1st_order_mass
    from pySDC.implementations.hooks.log_errors import LogGlobalErrorPostRun
    from pySDC.implementations.hooks.log_work import LogWork

    description = {
        'problem_class': FormFiredrakeIMEX if imex else FormFiredrake,
        'problem_params': get_forms(imex),
        'sweeper_class': imex_1st_order_mass if imex else generic_implicit_mass,
        'sweeper_params': {'quad_type': 'RADAU-RIGHT', 'num_nodes': 3, 'QI': 'LU'},
        'level_params': {'restol': 1e-10, 'dt': 0.1},
        'step_params': {'maxiter': 20},
    }
    controller_params = {'logger_level': 30, 'hook_class': [LogGlobalErrorPostRun, LogWork]}
    return description, controller_params


# %% [markdown]
# ## Running it
#
# The run prints the error at the end, the SDC iterations and the work of the problem class: how many solvers it set
# up (one per distinct diagonal entry of the preconditioner), how many systems it solved, and how many linear solves
# those took in total, including Newton's.


# %%
def run(imex):
    """
    Run the example with fully implicit or IMEX SDC

    Args:
        imex (bool): Whether to run IMEX SDC instead of fully implicit SDC

    Returns:
        dict: Error, iterations and work of the run
    """
    import numpy as np
    from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
    from pySDC.helpers.stats_helper import get_sorted

    description, controller_params = setup(imex)
    controller = controller_nonMPI(num_procs=1, controller_params=controller_params, description=description)

    P = controller.MS[0].levels[0].prob
    uend, stats = controller.run(u0=P.u_exact(0.0), t0=0.0, Tend=1.0)

    result = {'error': get_sorted(stats, type='e_global_post_run')[0][1]}
    for key in ['niter', 'work_solver_setup', 'work_solver', 'work_ksp']:
        result[key] = int(np.sum([me[1] for me in get_sorted(stats, type=key)]))

    variant = 'IMEX' if imex else 'implicit'
    summary = (
        f'{variant:8s} SDC: error {result["error"]:.2e}, {result["niter"]} iterations, '
        f'{result["work_solver_setup"]} solver setups, {result["work_solver"]} solves, '
        f'{result["work_ksp"]} linear solves'
    )
    print(summary)
    Path('data').mkdir(parents=True, exist_ok=True)
    with open('data/step_7_H_out.txt', 'a') as file:
        file.write(summary + '\n')
    return result


def main():
    """Run both variants and check that we got the same as last time."""
    implicit = run(imex=False)
    imex = run(imex=True)

    for result in [implicit, imex]:
        assert result['error'] < 1e-8, result
        assert result['work_solver_setup'] == 3, result
        assert result['work_solver'] == 3 * result['niter'], result

    # the reaction is mild enough that moving it to the explicit side costs no iterations here
    assert implicit['niter'] == imex['niter'] == 58, (implicit, imex)
    # the IMEX systems are linear: one linear solve each, while Newton needs a few
    assert imex['work_ksp'] == imex['work_solver'], imex
    assert implicit['work_ksp'] > implicit['work_solver'], implicit
    return implicit, imex


# %%
if __name__ == "__main__":
    main()

# %% [markdown]
# Both variants reach the same accuracy. IMEX SDC solves one linear system per node and iteration, where fully implicit
# SDC runs Newton's method. Which one is cheaper overall depends on how stiff the explicit part is: a stiffer reaction
# would cost IMEX SDC iterations, or a smaller step size.
#
# Firedrake does not run in the browser, nor in the environment this website is built in. Our CI runs this part in a
# Firedrake container and checks its errors and work counts. These are its results, in the run that built this page:
#
# :::{literalinclude} /../../data_firedrake/step_7_H_out.txt
# :language: text
# :::
