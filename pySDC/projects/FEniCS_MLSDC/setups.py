"""
Problem setups for the FEniCS mass-matrix reference project.

Every setup here takes the **mass-matrix route only**: ``eval_f`` returns the assembled weak form (a
load vector), ``solve_system`` takes a right-hand side that is already in the dual space, and
``base_transfer_mass`` restricts the FAS ``tau`` and ``u0`` with :math:`P^T`. Nothing anywhere
inverts the mass matrix.

Three examples, two element families and two coarsening directions:

* ``family='CG'`` -- continuous Lagrange elements, Dirichlet data imposed strongly.
* ``family='DG'`` -- discontinuous Lagrange elements, diffusion by interior penalty and Dirichlet
  data imposed weakly, see :mod:`problem_classes.DG_1D_FEniCS`.
* ``coarsening='h'`` -- coarsen the mesh, keep the element order.
* ``coarsening='p'`` -- keep the mesh, drop the element order.

Both directions give nested spaces for both families, so the same
:class:`mesh_to_mesh_fenics` transfer serves all four combinations. The collocation nodes are
deliberately kept on every level: a coarse level with fewer nodes is asymptotically inert at best,
and actively harmful in between.
"""

import math

from pySDC.implementations.problem_classes.HeatEquation_1D_FEniCS_matrix_forced import fenics_heat_mass
from pySDC.implementations.problem_classes.GrayScott_1D_FEniCS_implicit import fenics_grayscott_mass
from pySDC.implementations.problem_classes.VorticityVelocity_2D_FEniCS_periodic import fenics_vortex_2d_mass
from pySDC.projects.FEniCS_MLSDC.problem_classes.Burgers_1D_FEniCS import fenics_burgers_mass
from pySDC.projects.FEniCS_MLSDC.problem_classes.DG_1D_FEniCS import (
    fenics_heat_dg_mass,
    fenics_burgers_dg_mass,
    fenics_grayscott_dg_mass,
)
from pySDC.implementations.sweeper_classes.imex_1st_order_mass import imex_1st_order_mass
from pySDC.implementations.sweeper_classes.generic_implicit_mass import generic_implicit_mass
from pySDC.implementations.transfer_classes.BaseTransfer_mass import base_transfer_mass
from pySDC.implementations.transfer_classes.TransferFenicsMesh import mesh_to_mesh_fenics

EXAMPLES = ('heat', 'burgers', 'grayscott', 'vortex')
FAMILIES = ('CG', 'DG')
COARSENINGS = ('h', 'p')

#: Interior penalty constant for the DG forms. Only has to clear the coercivity threshold; the
#: multilevel behaviour is flat in it from 2 upwards.
SIGMA = 10.0

#: Per-example defaults. ``refinements`` (h-coarsening) and ``orders`` (p-coarsening) are indexed by
#: level, so the fine level is always the first entry; everything else is shared across levels. Both
#: ladders halve the dof count per level, so the two coarsening directions cost the same.
_SETUPS = {
    'heat': {
        'problem_class': fenics_heat_mass,
        'dg_problem_class': fenics_heat_dg_mass,
        'sweeper_class': imex_1st_order_mass,
        'problem_params': {'nu': 0.1, 't0': 0.0, 'c_nvars': 128, 'c': 1.0},
        'sweeper_params': {'quad_type': 'RADAU-RIGHT', 'QI': 'LU'},
        'refinements': [2, 1, 0],
        'orders': [4, 2, 1],
        'dt': 0.2,
        'nsteps': 8,
        'restol': 5e-10 / 500,
        'maxiter': 20,
        'num_nodes': 3,
        'utol': 1e-8,
        'pfasst_procs': (1, 2, 4, 8),
    },
    'grayscott': {
        'problem_class': fenics_grayscott_mass,
        'dg_problem_class': fenics_grayscott_dg_mass,
        'sweeper_class': generic_implicit_mass,
        'problem_params': {
            'c_nvars': 64,
            'newton_tol': 1e-12,
            'newton_rtol': 1e-11,
        },
        'sweeper_params': {'quad_type': 'RADAU-RIGHT', 'QI': 'LU'},
        'refinements': [2, 1, 0],
        'orders': [4, 2, 1],
        'dt': 1.0,
        'nsteps': 8,
        'restol': 1e-9,
        'maxiter': 30,
        'num_nodes': 3,
        'utol': 1e-6,
        'pfasst_procs': (1, 2, 4, 8),
    },
    'burgers': {
        'problem_class': fenics_burgers_mass,
        'dg_problem_class': fenics_burgers_dg_mass,
        'sweeper_class': generic_implicit_mass,
        'problem_params': {
            'c_nvars': 64,
            'nu': 0.02,
            'newton_tol': 1e-12,
            'newton_rtol': 1e-11,
        },
        'sweeper_params': {'quad_type': 'RADAU-RIGHT', 'QI': 'LU'},
        'refinements': [2, 1, 0],
        'orders': [4, 2, 1],
        'dt': 0.02,
        'nsteps': 8,
        'restol': 1e-9,
        'maxiter': 30,
        'num_nodes': 3,
        # PFASST on a nonlinear problem drifts from serial by more than the linear examples do
        'utol': 1e-6,
        'pfasst_procs': (1, 2, 4, 8),
    },
    # The vortex is the correctness example, not a savings example: MLSDC converges to the same
    # answer but the coarse level does not pay for itself, 0.73x on two levels, and four checks
    # failed to move that -- the preconditioner, the shear-layer width, the element order and a 16x
    # refinement of the whole hierarchy. It earns its place for what it covers rather than what it
    # saves: it is the only 2d and the only periodic hierarchy here, and while it was out of the
    # suite the prolongation was rewritten in a way that extrapolated on periodic spaces, with
    # nothing left to catch it. CG only, because the DG problem classes are 1d.
    'vortex': {
        'problem_class': fenics_vortex_2d_mass,
        'dg_problem_class': None,
        'sweeper_class': imex_1st_order_mass,
        'problem_params': {'nu': 0.01, 'delta': 0.05, 'rho': 50, 'c_nvars': [(8, 8)]},
        'sweeper_params': {'quad_type': 'RADAU-RIGHT', 'QI': 'LU'},
        'families': ('CG',),
        'coarsenings': ('h',),
        'refinements': [2, 1, 0],
        'orders': [2],
        'dt': 0.001,
        'nsteps': 8,
        'restol': 5e-9 / 500,
        'maxiter': 30,
        'num_nodes': 3,
        'utol': 1e-6,
        # The order study does not apply: it asks whether a high-order coarse space buys
        # savings, and this example has none to buy. Its base order is 2 anyway, so the CG4
        # rung would need a negative refinement.
        'order_study': (),
        'pays_off': False,
        # 8 steps used to report convergence and return an O(1) wrong answer here. That was the
        # point sampling: with it the error against serial is 1.21e+00 at 22 iterations, with the
        # L2 projection 3.9e-08 at 11.25.
        'pfasst_procs': (1, 2, 4, 8),
    },
}


def get_tolerance(example):
    """Tolerance for comparing solutions of the same example across configurations."""
    if example not in _SETUPS:
        raise ValueError(f'unknown example {example!r}, expected one of {EXAMPLES}')
    return _SETUPS[example]['utol']


def get_pfasst_procs(example):
    """Process counts PFASST is known to be reliable for on this example."""
    if example not in _SETUPS:
        raise ValueError(f'unknown example {example!r}, expected one of {EXAMPLES}')
    return _SETUPS[example]['pfasst_procs']


def get_families(example):
    """Element families this example has a problem class for. Not every example has both."""
    if example not in _SETUPS:
        raise ValueError(f'unknown example {example!r}, expected one of {EXAMPLES}')
    return _SETUPS[example].get('families', FAMILIES)


def get_coarsenings(example):
    """Coarsening directions this example is set up for."""
    if example not in _SETUPS:
        raise ValueError(f'unknown example {example!r}, expected one of {EXAMPLES}')
    return _SETUPS[example].get('coarsenings', COARSENINGS)


def pays_off(example):
    """Whether the coarse level is expected to cost less than it saves on this example."""
    if example not in _SETUPS:
        raise ValueError(f'unknown example {example!r}, expected one of {EXAMPLES}')
    return _SETUPS[example].get('pays_off', True)


def get_order_study(example):
    """Element orders to compare at equal dof counts, or an empty tuple where that has no meaning."""
    if example not in _SETUPS:
        raise ValueError(f'unknown example {example!r}, expected one of {EXAMPLES}')
    return _SETUPS[example].get('order_study', (1, 2, 4))


def get_description(
    example, nlevels=1, family='CG', coarsening='h', order=None, dt=None, restol=None, maxiter=None, nsteps=None
):
    """
    Build the pySDC description for one of the three examples.

    Parameters
    ----------
    example : str
        One of :data:`EXAMPLES`.
    nlevels : int
        1 for SDC, 2 or 3 for MLSDC/PFASST.
    family : str
        One of :data:`FAMILIES`: ``'CG'`` or ``'DG'``.
    coarsening : str
        One of :data:`COARSENINGS`: ``'h'`` coarsens the mesh at fixed element order, ``'p'`` drops
        the element order on the fine mesh. Irrelevant for ``nlevels=1``.
    order : int, optional
        Element order for h-coarsening, overriding the default. The refinement ladder is shifted to
        keep the dof count on every level, so that changing this isolates the *order* of the coarse
        space from its size: in 1d a Lagrange space has ``order * cells + 1`` dofs, so halving the
        order and doubling the cells leaves the count alone. Only meaningful with ``coarsening='h'``.
    dt, restol, maxiter, nsteps : optional
        Override the per-example defaults.

    Returns
    -------
    description : dict
    controller_params : dict
    t0, Tend : float
    """
    if example not in _SETUPS:
        raise ValueError(f'unknown example {example!r}, expected one of {EXAMPLES}')
    if family not in FAMILIES:
        raise ValueError(f'unknown family {family!r}, expected one of {FAMILIES}')
    if coarsening not in COARSENINGS:
        raise ValueError(f'unknown coarsening {coarsening!r}, expected one of {COARSENINGS}')
    if family not in get_families(example):
        raise ValueError(f'{example!r} has no {family} problem class, only {get_families(example)}')
    if coarsening not in get_coarsenings(example):
        raise ValueError(f'{example!r} is only set up for {get_coarsenings(example)}-coarsening')
    if not 1 <= nlevels <= 3:
        raise ValueError(f'nlevels must be 1, 2 or 3, got {nlevels}')

    s = _SETUPS[example]
    dt = s['dt'] if dt is None else dt
    nsteps = s['nsteps'] if nsteps is None else nsteps

    problem_params = dict(s['problem_params'], family=family)
    if coarsening == 'h':
        # per level: take the coarsest nlevels entries so the fine level stays fixed
        refinements = s['refinements'][:nlevels]
        element_order = s['orders'][0] if order is None else order
        if order is not None:
            # keep the dof count: halving the order and doubling the cells per dimension leaves it
            shift = round(math.log2(s['orders'][0] / order))
            refinements = [r + shift for r in refinements]
            if min(refinements) < 0:
                # dolfin just skips the refine loop, silently giving two identical levels
                raise ValueError(
                    f'{example!r} cannot hold its dof count at order {order}: that needs refinements '
                    f'{refinements}, and a negative refinement is not a coarsening'
                )
        problem_params['refinements'] = refinements
        problem_params['order'] = element_order
        orders = [element_order] * nlevels
    else:
        if order is not None:
            raise ValueError('the order override only applies to h-coarsening')
        # same mesh everywhere, the fine one; the element order is what comes down
        refinements = [s['refinements'][0]] * nlevels
        problem_params['refinements'] = s['refinements'][0]
        problem_params['order'] = s['orders'][:nlevels]
        orders = s['orders'][:nlevels]

    if family == 'DG':
        # Pin the interior penalty to the fine level. The SIPG form depends on the mesh and order it
        # is built on, so rediscretising it on a coarse level would change the penalty by h or p^2 --
        # and the penalty is the dominant term, so the coarse operator would stop being the Galerkin
        # operator P^T A_F P of the fine one. alpha_l = sigma p_0^2 h_l / h_0 keeps alpha_l / h_l
        # equal on every level, which restores A_G = P^T A_F P exactly.
        problem_params['penalty'] = [SIGMA * orders[0] ** 2 * 2 ** (refinements[0] - r) for r in refinements]

    sweeper_params = dict(s['sweeper_params'])
    sweeper_params['num_nodes'] = [s['num_nodes']] * nlevels

    description = {
        'problem_class': s['dg_problem_class'] if family == 'DG' else s['problem_class'],
        'problem_params': problem_params,
        'sweeper_class': s['sweeper_class'],
        'sweeper_params': sweeper_params,
        'level_params': {'restol': s['restol'] if restol is None else restol, 'dt': dt},
        'step_params': {'maxiter': s['maxiter'] if maxiter is None else maxiter},
    }

    if nlevels > 1:
        description['space_transfer_class'] = mesh_to_mesh_fenics
        description['space_transfer_params'] = {}
        description['base_transfer_class'] = base_transfer_mass
        # prolong_f is not available for the mass formulation: f is a load vector, so it cannot be
        # interpolated. base_transfer_mass falls back to prolong, re-evaluating f on the fine level.
        description['base_transfer_params'] = {'finter': False}

    controller_params = {'logger_level': 30}

    return description, controller_params, 0.0, nsteps * dt
