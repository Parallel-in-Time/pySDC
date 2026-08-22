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

from pySDC.implementations.problem_classes.HeatEquation_1D_FEniCS_matrix_forced import fenics_heat_mass
from pySDC.implementations.problem_classes.GrayScott_1D_FEniCS_implicit import fenics_grayscott_mass
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

EXAMPLES = ('heat', 'burgers', 'grayscott')
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


def get_description(example, nlevels=1, family='CG', coarsening='h', dt=None, restol=None, maxiter=None, nsteps=None):
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
    if not 1 <= nlevels <= 3:
        raise ValueError(f'nlevels must be 1, 2 or 3, got {nlevels}')

    s = _SETUPS[example]
    dt = s['dt'] if dt is None else dt
    nsteps = s['nsteps'] if nsteps is None else nsteps

    problem_params = dict(s['problem_params'], family=family)
    if coarsening == 'h':
        # per level: take the coarsest nlevels entries so the fine level stays fixed
        refinements = s['refinements'][:nlevels]
        problem_params['refinements'] = refinements
        problem_params['order'] = s['orders'][0]
        orders = [s['orders'][0]] * nlevels
    else:
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
