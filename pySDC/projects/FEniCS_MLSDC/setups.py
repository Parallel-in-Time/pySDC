"""
Problem setups for the FEniCS mass-matrix reference project.

Every setup here takes the **mass-matrix route only**: ``eval_f`` returns the assembled weak form (a
load vector), ``solve_system`` takes a right-hand side that is already in the dual space, and
``base_transfer_mass`` restricts the FAS ``tau`` and ``u0`` with :math:`P^T`. Nothing anywhere
inverts the mass matrix.

Coarsening is by mesh refinement. The collocation nodes are deliberately kept on every level: a
coarse level with fewer nodes is asymptotically inert at best, and actively harmful in between.
"""

from pySDC.implementations.problem_classes.HeatEquation_1D_FEniCS_matrix_forced import fenics_heat_mass
from pySDC.implementations.problem_classes.GrayScott_1D_FEniCS_implicit import fenics_grayscott_mass
from pySDC.implementations.problem_classes.VorticityVelocity_2D_FEniCS_periodic import fenics_vortex_2d_mass
from pySDC.implementations.sweeper_classes.imex_1st_order_mass import imex_1st_order_mass
from pySDC.implementations.sweeper_classes.generic_implicit_mass import generic_implicit_mass
from pySDC.implementations.transfer_classes.BaseTransfer_mass import base_transfer_mass
from pySDC.implementations.transfer_classes.TransferFenicsMesh import mesh_to_mesh_fenics

EXAMPLES = ('heat', 'grayscott', 'vortex')

#: Per-example defaults. ``refinements`` is indexed by level, so the fine level is always the first
#: entry; everything else is shared across levels.
_SETUPS = {
    'heat': {
        'problem_class': fenics_heat_mass,
        'sweeper_class': imex_1st_order_mass,
        'problem_params': {'nu': 0.1, 't0': 0.0, 'c_nvars': [128], 'family': 'CG', 'order': [4], 'c': 1.0},
        'sweeper_params': {'quad_type': 'RADAU-RIGHT'},
        'refinements': [2, 1, 0],
        'dt': 0.2,
        'nsteps': 8,
        'restol': 5e-10 / 500,
        'maxiter': 20,
        'num_nodes': 3,
        'utol': 1e-8,
    },
    'grayscott': {
        'problem_class': fenics_grayscott_mass,
        'sweeper_class': generic_implicit_mass,
        'problem_params': {
            'c_nvars': 64,
            'family': 'CG',
            'order': [4],
            'newton_tol': 1e-12,
            'newton_rtol': 1e-11,
        },
        'sweeper_params': {'quad_type': 'RADAU-RIGHT', 'QI': 'LU'},
        'refinements': [2, 1, 0],
        'dt': 1.0,
        'nsteps': 8,
        'restol': 1e-9,
        'maxiter': 30,
        'num_nodes': 3,
        'utol': 1e-6,
    },
    # The vortex is the correctness example, not a savings example: MLSDC converges to the same
    # answer here but the coarse level does not pay for itself. Measured insensitive to the layer
    # thickness (delta 0.05/0.15/0.3 all give the same counts) and to the preconditioner, so it is a
    # property of the problem, not of the setup. See the README.
    'vortex': {
        'problem_class': fenics_vortex_2d_mass,
        'sweeper_class': imex_1st_order_mass,
        'problem_params': {
            'nu': 0.01,
            'delta': 0.05,
            'rho': 50,
            'c_nvars': [(8, 8)],
            'family': 'CG',
            'order': [2],
        },
        'sweeper_params': {'quad_type': 'RADAU-RIGHT'},
        'refinements': [2, 1, 0],
        'dt': 0.001,
        'nsteps': 8,
        'restol': 5e-9 / 500,
        'maxiter': 30,
        'num_nodes': 3,
        'utol': 1e-6,
    },
}


def get_tolerance(example):
    """Tolerance for comparing solutions of the same example across configurations."""
    if example not in _SETUPS:
        raise ValueError(f'unknown example {example!r}, expected one of {EXAMPLES}')
    return _SETUPS[example]['utol']


def get_description(example, nlevels=1, dt=None, restol=None, maxiter=None, nsteps=None):
    """
    Build the pySDC description for one of the three examples.

    Parameters
    ----------
    example : str
        One of :data:`EXAMPLES`.
    nlevels : int
        1 for SDC, 2 or 3 for MLSDC/PFASST. Coarsening is by mesh refinement only.
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
    if not 1 <= nlevels <= 3:
        raise ValueError(f'nlevels must be 1, 2 or 3, got {nlevels}')

    s = _SETUPS[example]
    dt = s['dt'] if dt is None else dt
    nsteps = s['nsteps'] if nsteps is None else nsteps

    problem_params = dict(s['problem_params'])
    # refinements is per level: take the coarsest nlevels entries so the fine level stays fixed
    problem_params['refinements'] = s['refinements'][:nlevels]

    sweeper_params = dict(s['sweeper_params'])
    sweeper_params['num_nodes'] = [s['num_nodes']] * nlevels

    description = {
        'problem_class': s['problem_class'],
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
