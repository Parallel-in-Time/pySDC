r"""
Runnable demonstration of delta-form SDC and MLSDC with reduced-precision node-local solves.

Allen-Cahn, a nonlinear implicit operator, so both the node-local solve and the sweeper's own
increment go through the analytic expansions in :mod:`.problems`. Shows the claims the project
rests on, in one table:

1. the delta-form sweep reproduces :class:`generic_implicit` exactly, single- and multi-level;
2. running the node-local solve at ``float32`` neither changes the iteration count nor degrades the
   accuracy, because the solver is handed a *correction*;
3. the same argument applies to a whole coarse level, but only in delta form -- stock MLSDC stalls
   once its coarse level drops below backend precision, because it rebuilds the coarse residual and
   the coarse-grid correction by cancelling :math:`\mathcal{O}(1)` quantities.
"""

import numpy as np

from pySDC.helpers.stats_helper import get_sorted
from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.implementations.problem_classes.AllenCahn_2D_FD import allencahn_fullyimplicit
from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit
from pySDC.implementations.transfer_classes.TransferMesh import mesh_to_mesh
from pySDC.projects.DeltaSDC.cascade import delta_implicit_cascade
from pySDC.projects.DeltaSDC.mlsdc import (
    delta_implicit_rounded,
    delta_transfer,
    rounding_transfer,
)
from pySDC.projects.DeltaSDC.problems import allencahn_delta, heat_delta
from pySDC.implementations.sweeper_classes.delta_form import delta_implicit

SWEEPER_PARAMS = {
    'quad_type': 'RADAU-RIGHT',
    'node_type': 'LEGENDRE',
    'num_nodes': 3,
    'QI': 'LU',
    'initial_guess': 'spread',
}

BASE_PARAMS = {
    'nvars': (64, 64),
    'eps': 0.04,
    'newton_maxiter': 100,
    'newton_tol': 1e-12,
    # lin_tol is relative, so it can be very loose for free: at 1e-2 the accuracy is
    # unchanged and the linear work halves. newton_tol is absolute and must stay tight.
    'lin_tol': 1e-2,
    'lin_maxiter': 500,
    'radius': 0.25,
}

COARSE_NVARS = (32, 32)


def run(
    problem_class,
    problem_params,
    sweeper_class,
    sweeper_params,
    dt=4e-3,
    nsteps=2,
    restol=1e-9,
    multilevel=False,
    base_transfer_class=None,
):
    """
    Run a short Allen-Cahn simulation and return the end value together with diagnostics.

    Parameters
    ----------
    problem_class : type
        Problem class to integrate.
    problem_params : dict
        Parameters for the problem class.
    sweeper_class : type
        Sweeper class to use.
    sweeper_params : dict
        Parameters for the sweeper.
    dt : float, optional
        Step size.
    nsteps : int, optional
        Number of steps.
    restol : float, optional
        Residual tolerance for the SDC iteration.
    multilevel : bool, optional
        Add a coarse level, halving the resolution in each direction.
    base_transfer_class : type, optional
        Space-time transfer, defaulting to pySDC's :class:`BaseTransfer`.

    Returns
    -------
    dict
        Keys ``uend``, ``niter`` and ``work``.
    """
    description = {
        'problem_class': problem_class,
        'problem_params': problem_params,
        'sweeper_class': sweeper_class,
        'sweeper_params': sweeper_params,
        'level_params': {'restol': restol, 'dt': dt},
        'step_params': {'maxiter': 30},
    }
    if multilevel:
        description['problem_params'] = dict(problem_params, nvars=[BASE_PARAMS['nvars'], COARSE_NVARS])
        description['space_transfer_class'] = mesh_to_mesh
        description['space_transfer_params'] = {'iorder': 4, 'rorder': 2}
        if base_transfer_class is not None:
            description['base_transfer_class'] = base_transfer_class

    controller = controller_nonMPI(num_procs=1, controller_params={'logger_level': 30}, description=description)
    levels = controller.MS[0].levels
    uend, stats = controller.run(u0=levels[0].prob.u_exact(0.0), t0=0.0, Tend=nsteps * dt)

    work = {}
    for level in levels:
        for key, counter in level.prob.work_counters.items():
            work[key] = work.get(key, 0) + counter.niter
    return {
        'uend': uend,
        'niter': sum(value for _, value in get_sorted(stats, type='niter')),
        'work': work,
    }


def configurations():
    """
    The comparison matrix: single- and multi-level, stock and delta form, full and reduced precision.

    ``solve_precision`` is the node-local solve, ``level_precision`` everything a level stores. Both
    take the usual pySDC per-level list, so ``[None, np.float16]`` means "backend precision on the
    fine level, half precision on the coarse one".

    Returns
    -------
    list
        ``(label, problem_class, problem_params, sweeper_class, sweeper_params, run_kwargs)``.
    """
    delta_ml = {'multilevel': True, 'base_transfer_class': delta_transfer}
    return [
        ('SDC', allencahn_fullyimplicit, {}, generic_implicit, {}, {}),
        ('deltaSDC', allencahn_delta, {}, delta_implicit, {}, {}),
        ('fp32-deltaSDC', allencahn_delta, {'solve_precision': np.float32}, delta_implicit, {}, {}),
        ('MLSDC', allencahn_fullyimplicit, {}, generic_implicit, {}, {'multilevel': True}),
        ('deltaMLSDC', allencahn_delta, {}, delta_implicit_rounded, {}, delta_ml),
        ('fp32-deltaMLSDC', allencahn_delta, {'solve_precision': np.float32}, delta_implicit_rounded, {}, delta_ml),
        (
            # the coarse *solve* stays fp32 here: SciPy holds no float16 sparse matrix, and a
            # nonlinear correction solve in half precision additionally needs the unknown rescaled
            # to O(1), since fp16's smallest subnormal is 6e-8. The linear ladder below shows that.
            'fp16-coarse-deltaMLSDC',
            allencahn_delta,
            {'solve_precision': [np.float32, np.float32]},
            delta_implicit_rounded,
            {'level_precision': [None, np.float16]},
            delta_ml,
        ),
        # Storage precision raised as the iteration converges, rather than fixed: the fine level's
        # state is fp64 only at the *end*, and early sweeps neither need nor keep that. See mlsdc.py
        # for why the coarse level needs no such schedule -- its requirement is relative already.
        (
            'cascade fp16>fp32>fp64 fine',
            allencahn_delta,
            {'solve_precision': [np.float32, np.float32]},
            delta_implicit_cascade,
            {'level_precision': [None, np.float16], 'state_cascade': ('float16', 'float32', None)},
            delta_ml,
        ),
        # The controls, and they must break. The first says the delta *hierarchy* is what makes a
        # reduced-precision coarse level safe, not the node-local solve -- same sweeper, same
        # problem, only the residual and the coarse-grid correction handled the stock way. The
        # second says precision on the *fine* level binds, which is what makes every row above mean
        # something.
        (
            'CONTROL fp16 coarse, stock ML',
            allencahn_delta,
            {'solve_precision': [np.float32, np.float32]},
            delta_implicit_rounded,
            {'level_precision': [None, np.float16]},
            {'multilevel': True, 'base_transfer_class': rounding_transfer},
        ),
        (
            'CONTROL fp32 fine level',
            allencahn_delta,
            {},
            delta_implicit_rounded,
            {'level_precision': np.float32},
            delta_ml,
        ),
    ]


def main():
    """
    Run the comparison matrix and print it.

    Returns
    -------
    dict
        One result dictionary per configuration, keyed by label.
    """
    results = {}
    for label, problem_class, problem_extra, sweeper_class, sweeper_extra, run_kwargs in configurations():
        results[label] = run(
            problem_class,
            dict(BASE_PARAMS, **problem_extra),
            sweeper_class,
            dict(SWEEPER_PARAMS, **sweeper_extra),
            **run_kwargs,
        )

    peers = {label: 'MLSDC' if kwargs.get('multilevel') else 'SDC' for label, _, _, _, _, kwargs in configurations()}
    print(f"{'configuration':>30} | {'sweeps':>6} {'Newton':>7} {'CG':>7} | {'diff to SDC':>12} {'to fp64 peer':>13}")
    print('-' * 84)
    for label, result in results.items():
        diff = abs(result['uend'] - results['SDC']['uend'])
        peer = abs(result['uend'] - results[peers[label]]['uend'])
        print(
            f"{label:>30} | {result['niter']:>6} {result['work'].get('newton', 0):>7} "
            f"{result['work'].get('linear', 0):>7} | {diff:>12.3e} {peer:>13.3e}"
        )
    return results


HEAT_PARAMS = {
    'nvars': 127,
    'nu': 1.0,
    'freq': 2,
    'bc': 'dirichlet-zero',
    'order': 2,
    'solver_type': 'direct',
}

HEAT_COARSE_NVARS = 63


def run_heat(problem_params, sweeper_class, sweeper_params, multilevel, base_transfer_class=None, maxiter=25):
    """
    Run one step of the 1D heat equation to a fixed iteration count.

    Linear, so the delta form needs no problem-side solve: ``linear_implicit=True`` reaches the
    correction equation through the stock ``solve_system``. Run to a fixed iteration count rather
    than to a tolerance, so the residual reported is a floor rather than the tolerance that stopped
    the run.

    Parameters
    ----------
    problem_params : dict
        Parameters for :class:`heat_delta`.
    sweeper_class : type
        Sweeper class to use.
    sweeper_params : dict
        Parameters for the sweeper.
    multilevel : bool
        Add a coarse level at half the resolution.
    base_transfer_class : type, optional
        Space-time transfer, defaulting to pySDC's :class:`BaseTransfer`.
    maxiter : int, optional
        Number of iterations.

    Returns
    -------
    tuple
        The end value, the iteration at which the residual first fell below 1e-11 (or ``None``),
        and the residual floor reached.
    """
    description = {
        'problem_class': heat_delta,
        'problem_params': problem_params,
        'sweeper_class': sweeper_class,
        'sweeper_params': dict(sweeper_params, linear_implicit=True),
        'level_params': {'restol': -1, 'dt': 1e-1},
        'step_params': {'maxiter': maxiter},
    }
    if multilevel:
        description['problem_params'] = dict(problem_params, nvars=[HEAT_PARAMS['nvars'], HEAT_COARSE_NVARS])
        description['space_transfer_class'] = mesh_to_mesh
        description['space_transfer_params'] = {'iorder': 4, 'rorder': 2}
        if base_transfer_class is not None:
            description['base_transfer_class'] = base_transfer_class

    controller = controller_nonMPI(num_procs=1, controller_params={'logger_level': 30}, description=description)
    prob = controller.MS[0].levels[0].prob
    uend, stats = controller.run(u0=prob.u_exact(0.0), t0=0.0, Tend=1e-1)
    residuals = [value for _, value in get_sorted(stats, type='residual_post_iteration', sortby='iter')]
    hit = next((i + 1 for i, value in enumerate(residuals) if value < 1e-11), None)
    return uend, hit, min(residuals)


def heat_configurations():
    """
    The precision ladder on a linear problem, where every format is reachable.

    Half precision only shows up here: SciPy holds no ``float16`` sparse matrix, so the nonlinear
    Newton-CG route above cannot reach it without rescaling its unknown, while this direct solve can
    be emulated at any format.

    Returns
    -------
    list
        ``(label, problem_params, sweeper_class, sweeper_params, multilevel, transfer)``.
    """
    f32, f16 = np.float32, np.float16
    return [
        ('SDC', {}, delta_implicit, {}, False, None),
        ('fp32-deltaSDC', {'solve_precision': f32}, delta_implicit, {}, False, None),
        ('fp16-deltaSDC', {'solve_precision': f16}, delta_implicit, {}, False, None),
        ('MLSDC', {}, delta_implicit, {}, True, None),
        ('deltaMLSDC', {}, delta_implicit_rounded, {}, True, delta_transfer),
        ('fp32-coarse-solve', {'solve_precision': [None, f32]}, delta_implicit_rounded, {}, True, delta_transfer),
        ('fp16-coarse-solve', {'solve_precision': [None, f16]}, delta_implicit_rounded, {}, True, delta_transfer),
        (
            'fp16 coarse level and solve',
            {'solve_precision': [None, f16]},
            delta_implicit_rounded,
            {'level_precision': [None, f16]},
            True,
            delta_transfer,
        ),
        # Not emulated: the coarse level's arrays genuinely are float32 / float16, and its operators
        # follow. These are the rows the emulated ones above have to agree with.
        (
            'genuine fp32 coarse level',
            {'dtype': ['float64', 'float32']},
            delta_implicit_rounded,
            {},
            True,
            delta_transfer,
        ),
        (
            'genuine fp16 coarse level',
            {'dtype': ['float64', 'float16']},
            delta_implicit_rounded,
            {},
            True,
            delta_transfer,
        ),
        (
            # all three directions at once: a reduced fine solve, a coarse level reduced outright,
            # and the fine level's state climbing as the corrections shrink
            'all three: fp32 solve, fp16 coarse, cascade',
            {'solve_precision': [f32, f16]},
            delta_implicit_cascade,
            {'level_precision': [None, f16], 'state_cascade': ('float16', 'float32', None)},
            True,
            delta_transfer,
        ),
        (
            'full ladder: fp32 fine, fp16 coarse',
            {'solve_precision': [f32, f16]},
            delta_implicit_rounded,
            {'level_precision': [None, f16], 'correction_precision': [None, f16]},
            True,
            delta_transfer,
        ),
        # controls
        (
            'CONTROL fp16 coarse, stock ML',
            {'solve_precision': [None, f16]},
            delta_implicit_rounded,
            {'level_precision': [None, f16]},
            True,
            rounding_transfer,
        ),
        (
            'CONTROL fp16 solve, unnormalised',
            {'solve_precision': f16, 'normalize': False},
            delta_implicit,
            {},
            False,
            None,
        ),
        ('CONTROL fp32 fine level', {}, delta_implicit_rounded, {'level_precision': f32}, True, delta_transfer),
    ]


def main_heat():
    """
    Run the linear precision ladder and print it.

    Returns
    -------
    dict
        ``(uend, iterations, floor)`` per configuration, keyed by label.
    """
    results = {}
    for label, problem_extra, sweeper_class, sweeper_extra, multilevel, transfer in heat_configurations():
        results[label] = run_heat(
            dict(HEAT_PARAMS, **problem_extra),
            sweeper_class,
            dict(SWEEPER_PARAMS, **sweeper_extra),
            multilevel,
            transfer,
        )

    print(f"\n{'configuration':>36} | {'it to 1e-11':>11} {'floor':>10} | {'diff to SDC':>12}")
    print('-' * 76)
    for label, (uend, hit, floor) in results.items():
        diff = abs(uend - results['SDC'][0])
        print(f'{label:>36} | {str(hit):>11} {floor:>10.2e} | {diff:>12.3e}')
    return results


if __name__ == '__main__':
    main()
    main_heat()
