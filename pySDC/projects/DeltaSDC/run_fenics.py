r"""
Delta-form SDC on a FEniCS problem.

Runnable entry point in the style of ``pySDC/tutorial/step_7/A_pySDC_with_FEniCS.py``: the logic
lives here and the FEniCS-marked test simply calls :func:`main`.

Covers both routes on this backend:

* the **linear implicit** part, via the FEniCS heat equation and the IMEX delta form, which
  reproduces the stock ``imex_1st_order`` path exactly and needs no problem-class change;
* the **nonlinear implicit** part, via Gray-Scott and ``fenics_grayscott_delta``, which supplies a
  variational ``solve_system_delta`` with an analytically expanded increment.

Note what this backend showed about ``linear_implicit=True``: that shortcut reuses the stock
``solve_system`` to solve the correction equation directly, which additionally requires the solve to
be **homogeneous in its boundary conditions**. ``fenics_heat.solve_system`` applies inhomogeneous
Dirichlet data (``self.bc.apply(T, b)``) to whatever right-hand side it is handed, so using it for a
correction imposes the wrong boundary values -- the correction must carry *zero* boundary data. The
substitution fallback is used here instead, and is exact.

Reduced-precision *storage* of the corrections is **not** available on this backend: a
``fenics_mesh`` is backed by a DOLFIN function and cannot be built at another precision, so
requesting ``correction_precision`` raises a clear ``NotImplementedError``. That is asserted below
so the limitation stays visible. Reduced precision on a FEniCS backend would have to come from a
single-precision PETSc/DOLFIN build, which is a build-time choice.
"""

import numpy as np

from pySDC.helpers.stats_helper import get_sorted
from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.implementations.problem_classes.GrayScott_1D_FEniCS_implicit import fenics_grayscott
from pySDC.implementations.problem_classes.HeatEquation_1D_FEniCS_matrix_forced import fenics_heat
from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit
from pySDC.implementations.sweeper_classes.imex_1st_order import imex_1st_order
from pySDC.implementations.transfer_classes.TransferFenicsMesh import mesh_to_mesh_fenics
from pySDC.projects.DeltaSDC.cascade import delta_implicit_cascade
from pySDC.projects.DeltaSDC.mlsdc import (
    delta_imex_1st_order_rounded,
    delta_implicit_rounded,
    delta_transfer,
    rounding_transfer,
)
from pySDC.projects.DeltaSDC.problems_fenics import fenics_grayscott_delta, fenics_heat_no_increment
from pySDC.implementations.sweeper_classes.delta_form import delta_imex_1st_order, delta_implicit

T0 = 0.0

PROBLEM_PARAMS = {
    'nu': 0.1,
    't0': T0,
    'c_nvars': 128,
    'family': 'CG',
    'c': 1.0,
    'order': 4,
    'refinements': 1,
}


def sweeper_params(**extra):
    """Collocation and preconditioner settings, mirroring the step_7 tutorial."""
    params = {
        'quad_type': 'RADAU-RIGHT',
        'node_type': 'LEGENDRE',
        'num_nodes': 3,
        'QI': 'IE',
        'QE': 'EE',
        'initial_guess': 'spread',
    }
    params.update(extra)
    return params


def run(
    sweeper_class,
    sweeper_params_,
    maxiter=6,
    dt=0.2,
    nsteps=2,
    problem_class=fenics_heat,
    multilevel=False,
    base_transfer_class=None,
):
    """
    Run a short FEniCS heat-equation simulation.

    Parameters
    ----------
    sweeper_class : type
        Sweeper class to use.
    sweeper_params_ : dict
        Parameters for the sweeper.
    maxiter : int, optional
        Number of SDC iterations.
    dt : float, optional
        Step size.
    nsteps : int, optional
        Number of steps.
    problem_class : type, optional
        Problem class, which needs ``eval_f_increment`` for a reduced-precision coarse level.
    multilevel : bool, optional
        Add a coarse level, coarsened in the mesh and not in the element order or the nodes.
    base_transfer_class : type, optional
        Space-time transfer, defaulting to pySDC's :class:`BaseTransfer`.

    Returns
    -------
    dtype_u
        The end value.
    """
    description = {
        'problem_class': problem_class,
        'problem_params': PROBLEM_PARAMS,
        'sweeper_class': sweeper_class,
        'sweeper_params': sweeper_params_,
        'level_params': {'restol': -1, 'dt': dt},
        'step_params': {'maxiter': maxiter},
    }
    if multilevel:
        description['problem_params'] = dict(PROBLEM_PARAMS, refinements=[1, 0])
        description['space_transfer_class'] = mesh_to_mesh_fenics
        description['space_transfer_params'] = {}
        if base_transfer_class is not None:
            description['base_transfer_class'] = base_transfer_class
    controller = controller_nonMPI(num_procs=1, controller_params={'logger_level': 30}, description=description)
    prob = controller.MS[0].levels[0].prob
    uend, _ = controller.run(u0=prob.u_exact(T0), t0=T0, Tend=T0 + nsteps * dt)
    return uend


GRAYSCOTT_PARAMS = {
    'c_nvars': 64,
    't0': 0.0,
    'family': 'CG',
    # CG4 rather than CG2, because that is where the coarse level earns its keep: at dt=2 this
    # config takes 21 sweeps as SDC and 14 as MLSDC. With an inert coarse level the reduced-
    # precision rows below would prove nothing.
    'order': 4,
    'refinements': 1,
    'Du': 1.0,
    'Dv': 0.01,
    'A': 0.09,
    'B': 0.086,
    # Needed by the *stock* path in this comparison, whose unknown is the full state: its Newton
    # starts from a residual of order |u|, so a relative bar is a fixed absolute one and cannot
    # follow the SDC iteration down. The delta path needs neither -- its unknown is a correction, so
    # its relative bar tightens by itself -- but it is harmless to pass them to both.
    'newton_tol': 1e-13,
    'newton_rtol': 1e-13,
}


def run_grayscott(
    problem_class,
    problem_params,
    sweeper_class,
    sweeper_extra=None,
    multilevel=False,
    base_transfer_class=None,
    maxiter=30,
    dt=2.0,
    nsteps=2,
    restol=1e-10,
):
    """
    Run a short Gray-Scott simulation.

    Parameters
    ----------
    problem_class : type
        Problem class to integrate.
    problem_params : dict
        Parameters for the problem class.
    sweeper_class : type
        Sweeper class to use.
    maxiter : int, optional
        Number of SDC iterations.
    dt : float, optional
        Step size.
    nsteps : int, optional
        Number of steps.
    restol : float, optional
        Residual tolerance. Every configuration is compared at convergence, so a reduced-precision
        solve is allowed to take more sweeps but not to reach a different answer.

    Returns
    -------
    tuple
        The end value and the total number of sweeps.
    """
    description = {
        'problem_class': problem_class,
        'problem_params': problem_params,
        'sweeper_class': sweeper_class,
        'sweeper_params': dict(
            {
                'quad_type': 'RADAU-RIGHT',
                'node_type': 'LEGENDRE',
                'num_nodes': 3,
                'QI': 'LU',
                'initial_guess': 'spread',
            },
            **(sweeper_extra or {}),
        ),
        'level_params': {'restol': restol, 'dt': dt},
        'step_params': {'maxiter': maxiter},
    }
    if multilevel:
        description['problem_params'] = dict(problem_params, refinements=[1, 0])
        description['space_transfer_class'] = mesh_to_mesh_fenics
        description['space_transfer_params'] = {}
        if base_transfer_class is not None:
            description['base_transfer_class'] = base_transfer_class
    controller = controller_nonMPI(num_procs=1, controller_params={'logger_level': 40}, description=description)
    prob = controller.MS[0].levels[0].prob
    uend, stats = controller.run(u0=prob.u_exact(0.0), t0=0.0, Tend=nsteps * dt)
    return uend, sum(value for _, value in get_sorted(stats, type='niter'))


def heat_configurations():
    """
    The comparison matrix for the linear IMEX route.

    ``solve_precision`` does not appear here: the node-local solve goes through the substitution
    fallback on this problem, whose unknown is the full state, so reducing its precision is exactly
    the case the project says buys nothing. What a coarse level *can* do is drop the precision of
    everything it stores, which is what ``level_precision`` does.

    Returns
    -------
    list
        ``(label, problem_class, sweeper_class, sweeper_extra, run_kwargs)``.
    """
    delta_ml = {'multilevel': True, 'base_transfer_class': delta_transfer}
    stock_ml = {'multilevel': True, 'base_transfer_class': rounding_transfer}
    f32, f16 = np.float32, np.float16
    return [
        ('SDC', fenics_heat, imex_1st_order, {}, {}),
        ('deltaSDC', fenics_heat, delta_imex_1st_order, {}, {}),
        ('MLSDC', fenics_heat, imex_1st_order, {}, {'multilevel': True}),
        ('deltaMLSDC', fenics_heat, delta_imex_1st_order_rounded, {}, delta_ml),
        (
            'fp32-coarse-deltaMLSDC',
            fenics_heat,
            delta_imex_1st_order_rounded,
            {'level_precision': [None, f32]},
            delta_ml,
        ),
        (
            'fp16-coarse-deltaMLSDC',
            fenics_heat,
            delta_imex_1st_order_rounded,
            {'level_precision': [None, f16]},
            delta_ml,
        ),
        # No cascade row here on purpose. This table runs a fixed six iterations rather than to a
        # tolerance, and a precision cascade needs the run to converge: it climbs its ladder as the
        # corrections shrink, so a run cut short ends *while still in a low format* and returns that
        # answer -- 4.0e-07 out, when tried. The Gray-Scott table below runs to a tolerance and
        # carries the row instead.
        # Controls. The middle one is specific to this backend: |f| = |M^-1 K u| is of order 1e5
        # here, so forming the increment by subtraction loses 7.7e-11 *in double precision* already,
        # and a reduced-precision level multiplies that by 1/eps.
        (
            'CONTROL fp32 coarse, stock ML',
            fenics_heat,
            delta_imex_1st_order_rounded,
            {'level_precision': [None, f32]},
            stock_ml,
        ),
        (
            'CONTROL fp32 coarse, no increment',
            fenics_heat_no_increment,
            delta_imex_1st_order_rounded,
            {'level_precision': [None, f32]},
            delta_ml,
        ),
        ('CONTROL fp32 fine level', fenics_heat, delta_imex_1st_order_rounded, {'level_precision': f32}, delta_ml),
    ]


def check_linear():
    """
    Run the linear IMEX matrix and check it.

    Returns
    -------
    dict
        The end value per configuration, keyed by label.

    Raises
    ------
    AssertionError
        If a configuration deviates from its own full-precision counterpart, or a control does not
        break.
    """
    results = {}
    for label, problem_class, sweeper_class, sweeper_extra, run_kwargs in heat_configurations():
        results[label] = run(sweeper_class, sweeper_params(**sweeper_extra), problem_class=problem_class, **run_kwargs)

    print(f"{'configuration':>36} | {'diff to SDC':>12} {'to fp64 peer':>13}")
    print('-' * 66)
    for label, uend in results.items():
        peer = results['MLSDC' if 'ML' in label else 'SDC']
        deviation = abs(uend - peer)
        print(f'{label:>36} | {abs(uend - results["SDC"]):>12.3e} {deviation:>13.3e}')
        if label.startswith('CONTROL'):
            assert deviation > 1e-9, f'{label} did not break, so it no longer controls anything'
        else:
            assert deviation < 1e-10, f'{label} deviates from its full-precision peer by {deviation:.3e}'
    return results


def grayscott_configurations():
    """
    The comparison matrix for the nonlinear variational route.

    Returns
    -------
    list
        ``(label, problem_class, problem_extra, sweeper_class, sweeper_extra, run_kwargs)``.
    """
    delta_ml = {'multilevel': True, 'base_transfer_class': delta_transfer}
    f32, f16 = np.float32, np.float16
    return [
        ('SDC', fenics_grayscott, {}, generic_implicit, {}, {}),
        ('deltaSDC', fenics_grayscott_delta, {}, delta_implicit, {}, {}),
        ('fp32-deltaSDC', fenics_grayscott_delta, {'solve_precision': f32}, delta_implicit, {}, {}),
        ('MLSDC', fenics_grayscott, {}, generic_implicit, {}, {'multilevel': True}),
        ('deltaMLSDC', fenics_grayscott_delta, {}, delta_implicit_rounded, {}, delta_ml),
        ('fp32-deltaMLSDC', fenics_grayscott_delta, {'solve_precision': f32}, delta_implicit_rounded, {}, delta_ml),
        (
            'fp16-coarse-deltaMLSDC',
            fenics_grayscott_delta,
            {'solve_precision': [f32, f16]},
            delta_implicit_rounded,
            {'level_precision': [None, f16]},
            delta_ml,
        ),
        (
            'cascade fp16>fp32>fp64 fine',
            fenics_grayscott_delta,
            {'solve_precision': [f32, f16]},
            delta_implicit_cascade,
            {'level_precision': [None, f16], 'state_cascade': ('float16', 'float32', None)},
            delta_ml,
        ),
        (
            'CONTROL fp16 coarse, stock ML',
            fenics_grayscott_delta,
            {'solve_precision': [f32, f16]},
            delta_implicit_rounded,
            {'level_precision': [None, f16]},
            {'multilevel': True, 'base_transfer_class': rounding_transfer},
        ),
        # Half precision cannot represent what the delta form hands the solver unless the unknown is
        # scaled: fp16's smallest subnormal is 6e-8 and the correction falls below that.
        (
            'CONTROL fp16 solve, unnormalised',
            fenics_grayscott_delta,
            {'solve_precision': [f32, f16], 'normalize': False},
            delta_implicit_rounded,
            {'level_precision': [None, f16]},
            delta_ml,
        ),
    ]


def check_nonlinear():
    """
    Run the nonlinear Gray-Scott matrix and check it.

    With the node-local Newton tightened past the SDC residual tolerance, the agreement level is set
    by the SDC iteration rather than by the node-local solve, so it can be asserted much harder.

    Returns
    -------
    dict
        The end value per configuration, keyed by label.

    Raises
    ------
    AssertionError
        If a configuration deviates from its own full-precision counterpart, or a control does not
        break.
    """
    results = {}
    for label, problem_class, problem_extra, sweeper_class, sweeper_extra, run_kwargs in grayscott_configurations():
        results[label] = run_grayscott(
            problem_class,
            dict(GRAYSCOTT_PARAMS, **problem_extra),
            sweeper_class,
            sweeper_extra,
            **run_kwargs,
        )

    print(f"\n{'configuration':>36} | {'sweeps':>6} | {'diff to SDC':>12} {'to fp64 peer':>13}")
    print('-' * 76)
    for label, (uend, sweeps) in results.items():
        peer, _ = results['MLSDC' if 'ML' in label else 'SDC']
        deviation = abs(uend - peer)
        print(f'{label:>36} | {sweeps:>6} | {abs(uend - results["SDC"][0]):>12.3e} {deviation:>13.3e}')
        if label.startswith('CONTROL'):
            assert deviation > 1e-9, f'{label} did not break, so it no longer controls anything'
        else:
            assert deviation < 1e-9, f'{label} deviates from its full-precision peer by {deviation:.3e}'
    assert results['MLSDC'][1] < results['SDC'][1], 'the coarse level must earn its keep'
    return results


def main():
    """
    Check the delta form against the stock sweepers on a FEniCS backend, single- and multi-level.

    Returns
    -------
    dict
        The end values of the linear matrix, keyed by configuration.

    Raises
    ------
    AssertionError
        If any delta-form variant deviates from the stock path.
    """
    results = check_linear()

    # reduced-precision correction *storage* is unavailable here, and must say so clearly
    try:
        run(delta_imex_1st_order, sweeper_params(correction_precision=np.dtype('float32')))
    except NotImplementedError as error:
        print(f'correction_precision correctly refused: {error}')
    else:  # pragma: no cover - guards against a silent regression
        raise AssertionError('correction_precision should not be silently accepted for fenics_mesh')

    check_nonlinear()

    return results


if __name__ == '__main__':
    main()
