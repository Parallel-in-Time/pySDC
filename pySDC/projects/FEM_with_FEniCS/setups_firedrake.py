"""
The setups of :mod:`pySDC.projects.FEM_with_FEniCS.setups`, with the Firedrake ports of the problem classes.

The per-example settings are those of the FEniCS setups, except for the residual tolerance. FEniCS measured the
residual, a load vector, by the L2 norm of its entries read as a function; Firedrake measures it in the norm dual to
L2. :func:`get_description` scales the FEniCS tolerance by the ratio of the two norms on the fine level, which
:func:`norm_ratio` measures, so that the runs stop at comparable accuracy.

Meshes come from Firedrake ``MeshHierarchy`` objects (h-coarsening) or are shared by all levels (p-coarsening, lower
degrees). The 1D meshes are the same as FEniCS', the vortex's are not: DOLFIN's ``refine`` triangulates differently.
"""

import math

import firedrake as fd

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

#: The settings of the FEniCS setups. `restol` is the FEniCS tolerance, in the norm of the entries.
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
        'restol': 5e-10 / 500,
        'maxiter': 20,
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
        'restol': 1e-9,
        'maxiter': 30,
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
        'restol': 1e-9,
        'maxiter': 30,
        'order_study': (1, 2, 4),
    },
    'vortex': {
        'classes': {'CG': Vortex2DFiredrake},
        'sweeper_class': imex_1st_order_mass,
        'problem_params': {'nu': 0.01, 'delta': 0.05, 'rho': 50},
        'base_mesh': lambda: fd.PeriodicUnitSquareMesh(8, 8, diagonal='right'),
        'refinements': [2, 1, 0],
        'orders': [2],
        'coarsenings': ('h',),
        'dt': 0.001,
        'nsteps': 8,
        'restol': 5e-9 / 500,
        'maxiter': 30,
        'order_study': (),
    },
}


def get_families(example):
    """Element families this example has a problem class for."""
    return tuple(SETUPS[example]['classes'])


def get_coarsenings(example):
    """Coarsening directions this example is set up for."""
    return SETUPS[example].get('coarsenings', COARSENINGS)


def get_order_study(example):
    """Element orders to compare at equal dof counts, or empty where that has no meaning."""
    return SETUPS[example]['order_study']


def get_description(example, nlevels=1, family='CG', coarsening='h', order=None, restol=None, nsteps=None):
    """
    The pySDC description of one configuration, as :func:`pySDC.projects.FEM_with_FEniCS.setups.get_description`.

    Args:
        example (str): One of :data:`EXAMPLES`
        nlevels (int): 1 for SDC, 2 or 3 for MLSDC/PFASST
        family (str): 'CG' or 'DG'
        coarsening (str): 'h' for coarser meshes at fixed degree, 'p' for lower degrees on the fine mesh
        order (int, optional): Degree for h-coarsening, with the refinements shifted to keep the dofs per level
        restol (float, optional): Residual tolerance in the dual norm. By default the FEniCS one, scaled by
            :func:`norm_ratio`
        nsteps (int, optional): Number of steps

    Returns:
        description (dict), controller_params (dict), t0 (float), Tend (float)
    """
    s = SETUPS[example]
    if coarsening == 'h':
        refinements = list(s['refinements'][:nlevels])
        degree = s['orders'][0] if order is None else order
        if order is not None:
            # keep the dof count: halving the degree and doubling the cells leaves it alone in 1D
            refinements = [r + round(math.log2(s['orders'][0] / order)) for r in refinements]
        hierarchy = fd.MeshHierarchy(s['base_mesh'](), max(refinements))
        meshes = [hierarchy[r] for r in refinements]
        orders = [degree] * nlevels
        transfer = MeshToMeshFiredrakeHierarchy
    else:
        fine = fd.MeshHierarchy(s['base_mesh'](), s['refinements'][0])[-1]
        refinements = [s['refinements'][0]] * nlevels
        meshes = [fine] * nlevels
        orders = list(s['orders'][:nlevels])
        transfer = MeshToMeshFiredrake

    problem_params = dict(s['problem_params'], mesh=meshes, order=orders)
    if family == 'DG':
        # pin the interior penalty to the fine level, alpha_l / h_l equal on every level, see the FEniCS setups
        problem_params['penalty'] = [SIGMA * orders[0] ** 2 * 2 ** (refinements[0] - r) for r in refinements]

    if restol is None:
        restol = s['restol'] * norm_ratio(example, family)
    description = {
        'problem_class': s['classes'][family],
        'problem_params': problem_params,
        'sweeper_class': s['sweeper_class'],
        'sweeper_params': {'quad_type': 'RADAU-RIGHT', 'QI': 'LU', 'num_nodes': [3] * nlevels},
        'level_params': {'restol': restol, 'dt': s['dt']},
        'step_params': {'maxiter': s['maxiter']},
    }
    if nlevels > 1:
        description['space_transfer_class'] = transfer
        description['base_transfer_class'] = base_transfer_mass
        description['base_transfer_params'] = {'finter': False}
    nsteps = s['nsteps'] if nsteps is None else nsteps
    return description, {'logger_level': 30}, 0.0, nsteps * s['dt']


_RATIOS = {}


def norm_ratio(example, family):
    """
    Ratio of the dual norm to the FEniCS norm of a residual on the fine level, measured once per example and family on
    the residual after two SDC iterations of the first step.

    Returns:
        float: |r|_dual / |r|_FEniCS
    """
    from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI

    if (example, family) not in _RATIOS:
        description, controller_params, t0, _ = get_description(example, family=family, restol=-1.0)
        description['step_params']['maxiter'] = 2
        controller = controller_nonMPI(num_procs=1, controller_params=controller_params, description=description)
        L = controller.MS[0].levels[0]
        P = L.prob
        controller.run(u0=P.u_exact(t0), t0=t0, Tend=t0 + SETUPS[example]['dt'])
        res = L.sweep.integrate()[-1] + P.apply_mass_matrix(L.u[0] - L.u[-1])
        P.fix_residual(res)
        entries = fd.Function(P.V)
        with res.dat.vec_ro as source, entries.dat.vec_wo as target:
            source.copy(target)
        _RATIOS[(example, family)] = abs(res) / fd.norm(entries)
    return _RATIOS[(example, family)]
