"""
The setups of :mod:`pySDC.projects.FEM_with_FEniCS.setups`, with the Firedrake ports of the problem classes.

The per-example settings are those of the FEniCS setups, except for the residual tolerances, which are set in the
norm the Firedrake data types measure residuals in, the norm dual to L2. That norm does not depend on the mesh, so one
tolerance per example serves every family, element order and level.

Meshes come from Firedrake ``MeshHierarchy`` objects (h-coarsening) or are shared by all levels (p-coarsening, lower
degrees), with Firedrake's default triangulation and Lagrange nodes.
"""

import math

import firedrake as fd

from pySDC.implementations.convergence_controller_classes.spatial_error_floor_firedrake import (
    SpatialErrorFloorFiredrake,
)
from pySDC.implementations.problem_classes.GrayScott_1D_Firedrake import GrayScott1DFiredrake
from pySDC.implementations.problem_classes.HeatEquation_1D_FormFiredrake import Heat1DMassFiredrake
from pySDC.implementations.problem_classes.VorticityVelocity_2D_Firedrake_periodic import Vortex2DFiredrake
from pySDC.implementations.sweeper_classes.generic_implicit_mass import generic_implicit_mass
from pySDC.implementations.sweeper_classes.imex_1st_order_mass import imex_1st_order_mass
from pySDC.implementations.transfer_classes.BaseTransfer_mass import base_transfer_mass
from pySDC.implementations.transfer_classes.TransferFiredrakeMesh import (
    MeshToMeshFiredrake,
    MeshToMeshFiredrakeHierarchy,
)
from pySDC.projects.FEM_with_FEniCS.problem_classes.Burgers_1D_Firedrake import Burgers1DFiredrake
from pySDC.projects.FEM_with_FEniCS.problem_classes.DG_1D_Firedrake import (
    Burgers1DDGFiredrake,
    GrayScott1DDGFiredrake,
    Heat1DDGFiredrake,
)

EXAMPLES = ('heat', 'burgers', 'grayscott', 'vortex')
FAMILIES = ('CG', 'DG')
COARSENINGS = ('h', 'p')
SIGMA = 10.0
NEWTON = {'snes_atol': 1e-12, 'snes_rtol': 1e-11}

#: The settings of the examples. `restol` is the residual tolerance in the dual norm.
SETUPS = {
    'heat': {
        'classes': {'CG': Heat1DMassFiredrake, 'DG': Heat1DDGFiredrake},
        'sweeper_class': imex_1st_order_mass,
        'problem_params': {'nu': 0.1, 'c': 1.0},
        'base_mesh': lambda: fd.UnitIntervalMesh(128),
        'refinements': [2, 1, 0],
        'orders': [4, 2, 1],
        'dt': 0.2,
        'nsteps': 8,
        'restol': 2e-9,
        'maxiter': 20,
        'utol': 1e-8,
        'order_study': (1, 2, 4),
    },
    'burgers': {
        'classes': {'CG': Burgers1DFiredrake, 'DG': Burgers1DDGFiredrake},
        'sweeper_class': generic_implicit_mass,
        'problem_params': {'nu': 0.02, 'solver_parameters': NEWTON},
        'base_mesh': lambda: fd.UnitIntervalMesh(64),
        'refinements': [2, 1, 0],
        'orders': [4, 2, 1],
        'dt': 0.02,
        'nsteps': 8,
        'restol': 1e-6,
        'maxiter': 30,
        'utol': 1e-6,
        'order_study': (1, 2, 4),
    },
    'grayscott': {
        'classes': {'CG': GrayScott1DFiredrake, 'DG': GrayScott1DDGFiredrake},
        'sweeper_class': generic_implicit_mass,
        'problem_params': {'solver_parameters': NEWTON},
        'base_mesh': lambda: fd.IntervalMesh(64, 100),
        'refinements': [2, 1, 0],
        'orders': [4, 2, 1],
        'dt': 1.0,
        'nsteps': 8,
        'restol': 1e-8,
        'maxiter': 30,
        'utol': 1e-6,
        'order_study': (1, 2, 4),
    },
    'vortex': {
        'classes': {'CG': Vortex2DFiredrake},
        'sweeper_class': imex_1st_order_mass,
        'problem_params': {'nu': 0.01, 'delta': 0.05, 'rho': 50},
        'base_mesh': lambda: fd.PeriodicUnitSquareMesh(8, 8),
        'refinements': [2, 1, 0],
        'orders': [2],
        'coarsenings': ('h',),
        'dt': 0.001,
        'nsteps': 8,
        'restol': 1e-7,
        'maxiter': 30,
        'utol': 1e-6,
        'order_study': (),
        'pays_off': False,
    },
}


PFASST_PROCS = (1, 2, 4, 8)


def _setup(example):
    """The settings of one example, see :data:`SETUPS`."""
    if example not in SETUPS:
        raise ValueError(f'unknown example {example!r}, expected one of {EXAMPLES}')
    return SETUPS[example]


def get_families(example):
    """Element families this example has a problem class for."""
    return tuple(_setup(example)['classes'])


def get_coarsenings(example):
    """Coarsening directions this example is set up for."""
    return _setup(example).get('coarsenings', COARSENINGS)


def get_order_study(example):
    """Element orders to compare at equal dof counts, or empty where that has no meaning."""
    return _setup(example)['order_study']


def get_tolerance(example):
    """Tolerance for comparing solutions of the same example across configurations, in the L2 norm."""
    return _setup(example)['utol']


def get_pfasst_procs(example):
    """Process counts PFASST is run on for this example."""
    _setup(example)
    return PFASST_PROCS


def pays_off(example):
    """Whether the coarse level is expected to cost less than it saves on this example."""
    return _setup(example).get('pays_off', True)


def get_description(
    example,
    nlevels=1,
    family='CG',
    coarsening='h',
    order=None,
    dt=None,
    restol=None,
    maxiter=None,
    nsteps=None,
    spatial_error_floor=False,
):
    """
    The pySDC description of one configuration, as :func:`pySDC.projects.FEM_with_FEniCS.setups.get_description`.

    Args:
        example (str): One of :data:`EXAMPLES`
        nlevels (int): 1 for SDC, 2 or 3 for MLSDC/PFASST
        family (str): 'CG' or 'DG'
        coarsening (str): 'h' for coarser meshes at fixed degree, 'p' for lower degrees on the fine mesh
        order (int, optional): Degree for h-coarsening, with the refinements shifted to keep the dofs per level
        dt (float, optional): Step size
        restol (float, optional): Residual tolerance in the dual norm, by default the example's
        maxiter (int, optional): Maximum number of iterations
        nsteps (int, optional): Number of steps
        spatial_error_floor (bool, optional): Also stop a step once its residual is below its share of the spatial
            error, estimated on the next coarser mesh, with
            :class:`~pySDC.implementations.convergence_controller_classes.spatial_error_floor_firedrake.SpatialErrorFloorFiredrake`.
            CG only, and for SDC and MLSDC, not PFASST

    Returns:
        description (dict), controller_params (dict), t0 (float), Tend (float)
    """
    s = _setup(example)
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
    if coarsening == 'h':
        refinements = list(s['refinements'][:nlevels])
        degree = s['orders'][0] if order is None else order
        if order is not None:
            # keep the dof count: halving the degree and doubling the cells leaves it alone in 1D
            refinements = [r + round(math.log2(s['orders'][0] / order)) for r in refinements]
            if min(refinements) < 0:
                # a negative index would silently pick a level from the fine end of the hierarchy
                raise ValueError(
                    f'{example!r} cannot hold its dof count at order {order}: that needs refinements '
                    f'{refinements}, and a negative refinement is not a coarsening'
                )
        hierarchy = fd.MeshHierarchy(s['base_mesh'](), max(refinements))
        meshes = [hierarchy[r] for r in refinements]
        orders = [degree] * nlevels
        transfer = MeshToMeshFiredrakeHierarchy
    else:
        if order is not None:
            raise ValueError('the order override only applies to h-coarsening')
        hierarchy = fd.MeshHierarchy(s['base_mesh'](), s['refinements'][0])
        fine = hierarchy[-1]
        refinements = [s['refinements'][0]] * nlevels
        meshes = [fine] * nlevels
        orders = list(s['orders'][:nlevels])
        transfer = MeshToMeshFiredrake

    problem_params = dict(s['problem_params'], mesh=meshes, order=orders)
    if family == 'DG':
        # pin the interior penalty to the fine level, alpha_l / h_l equal on every level, see the FEniCS setups
        problem_params['penalty'] = [SIGMA * orders[0] ** 2 * 2 ** (refinements[0] - r) for r in refinements]

    if restol is None:
        restol = s['restol']
    description = {
        'problem_class': s['classes'][family],
        'problem_params': problem_params,
        'sweeper_class': s['sweeper_class'],
        'sweeper_params': {'quad_type': 'RADAU-RIGHT', 'QI': 'LU', 'num_nodes': [3] * nlevels},
        'level_params': {'restol': restol, 'dt': s['dt'] if dt is None else dt},
        'step_params': {'maxiter': s['maxiter'] if maxiter is None else maxiter},
    }
    if nlevels > 1:
        description['space_transfer_class'] = transfer
        description['base_transfer_class'] = base_transfer_mass
        description['base_transfer_params'] = {'finter': False}
    nsteps = s['nsteps'] if nsteps is None else nsteps
    Tend = nsteps * description['level_params']['dt']
    if spatial_error_floor:
        if family != 'CG':
            # shortcut: the DG classes need their interior penalty pinned for the coarse space, add it if DG needs this
            raise ValueError('the spatial error floor is set up for CG only')
        if refinements[0] < 1:
            raise ValueError(f'the spatial error floor needs a coarser mesh than refinement {refinements[0]}')
        # the fine level's space, built by the problem class on the next coarser mesh of the same hierarchy
        coarse_mesh = hierarchy[refinements[0] - 1]
        coarse_space = s['classes'][family](mesh=coarse_mesh, order=orders[0], **s['problem_params']).V
        description['convergence_controllers'] = {
            SpatialErrorFloorFiredrake: {'coarse_space': coarse_space, 'Tend': Tend}
        }
    return description, {'logger_level': 30}, 0.0, Tend
