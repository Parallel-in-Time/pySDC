"""
ParaDiag inherits the delta-form property without being reformulated.

Every test here compares an attainable *residual floor*, not a single answer: the claim is about
what precision caps, and a cap is only visible once the iteration has stopped making progress.

The last test is the control, and the file is worth little without it. The three before it assert
that reducing a precision changes nothing, which is exactly what a broken measurement also reports.
``test_reduced_precision_state_does_cap_accuracy`` reduces a precision *outside* the preconditioner
and asserts the floor rises by orders of magnitude, so a harness that had stopped measuring anything
fails there.
"""

import pytest

L_STEPS = 8
DT = 0.05
ALPHA = 1e-4
FLOOR_FULL = 1e-13
"""What double precision reaches on this problem, with margin. Measured value is ~6e-16."""


def _description(solve_precision=None):
    from pySDC.implementations.sweeper_classes.ParaDiagSweepers import QDiagonalizationIMEX
    from pySDC.projects.DeltaSDC.paradiag import heat_paradiag

    return {
        'problem_class': heat_paradiag,
        'problem_params': {
            'nu': 0.1,
            'freq': (2,),
            'nvars': (31,),
            'bc': 'dirichlet-zero',
            'solve_precision': solve_precision,
        },
        'sweeper_class': QDiagonalizationIMEX,
        'sweeper_params': {'quad_type': 'RADAU-RIGHT', 'num_nodes': 3},
        'level_params': {'restol': 1e-14, 'dt': DT},
        'step_params': {'maxiter': 30},
    }


def run(solve_precision=None, transform_precision=None, controller_class=None):
    """
    Run one ParaDiag configuration and report what it reached.

    Returns:
        tuple: the smallest residual seen, and the largest per-step iteration count
    """
    from pySDC.helpers.stats_helper import get_sorted
    from pySDC.projects.DeltaSDC.paradiag import controller_ParaDiag_reduced_transform

    controller_class = controller_ParaDiag_reduced_transform if controller_class is None else controller_class
    controller_params = {'logger_level': 40, 'alpha': ALPHA}
    if transform_precision is not None:
        controller_params['transform_precision'] = transform_precision

    controller = controller_class(
        num_procs=L_STEPS, controller_params=controller_params, description=_description(solve_precision)
    )
    P = controller.MS[0].levels[0].prob
    _, stats = controller.run(u0=P.u_exact(0), t0=0, Tend=L_STEPS * DT)

    residuals = [v for _, v in get_sorted(stats, type='residual_post_iteration', sortby='iter')]
    niter = [v for _, v in get_sorted(stats, type='niter', sortby='time')]
    return min(residuals), max(niter)


@pytest.mark.base
def test_paradiag_solves_for_an_increment_that_shrinks():
    """
    The property everything else rests on, asserted directly.

    ParaDiag hands the node-local solver the residual of the composite collocation problem and gets
    back an increment, which it adds to the solution. Nothing in this project arranges that -- it is
    what the ParaDiag controller already does -- so this is a characterisation test: if the
    controller ever stops solving for an increment, reduced precision stops being safe and the tests
    below would keep passing for the wrong reason.
    """
    import numpy as np
    from pySDC.projects.DeltaSDC.paradiag import controller_ParaDiag_reduced_transform

    magnitudes = []

    class Recorder(controller_ParaDiag_reduced_transform):
        def update_solution(self, local_MS_running):
            increment = max(
                float(abs(np.asarray(S.levels[0].increment[m])).max())
                for S in local_MS_running
                for m in range(S.levels[0].sweep.coll.num_nodes)
            )
            magnitudes.append(increment)
            super().update_solution(local_MS_running)

    run(controller_class=Recorder)

    assert len(magnitudes) > 2, 'need several iterations to say anything about the trend'
    assert magnitudes[-1] < 1e-6 * magnitudes[0], (
        'the increment must collapse as the iteration converges, otherwise a reduced-precision '
        f'solve introduces an error that does not vanish -- got {magnitudes}'
    )


@pytest.mark.base
def test_reduced_precision_solve_does_not_cap_accuracy():
    """The node-local solve at complex64 must still reach the double-precision floor."""
    full, _ = run()
    reduced, _ = run(solve_precision='complex64')

    assert full < FLOOR_FULL, f'the double-precision baseline itself did not converge, got {full:.3e}'
    assert reduced < FLOOR_FULL, (
        f'a complex64 node-local solve capped the residual at {reduced:.3e}, against {full:.3e} in '
        'full precision -- the increment formulation is supposed to prevent exactly this'
    )


@pytest.mark.base
def test_reduced_precision_transform_does_not_cap_accuracy():
    """
    The weighted FFT across the steps at complex64 must also still reach the floor.

    This is the surprising half. The transform is deliberately ill-conditioned -- the weights span
    ``alpha**(-(L-1)/L)``, here about 1e4 -- so the expectation is a floor near ``eps/alpha``, which
    for complex64 would be about 1e-3. It does not happen, because the amplification acts on the
    increment rather than on the solution.
    """
    full, _ = run()
    reduced, _ = run(transform_precision='complex64')

    assert reduced < FLOOR_FULL, (
        f'a complex64 weighted transform capped the residual at {reduced:.3e}, against {full:.3e} '
        f'in full precision (eps/alpha would be ~{6e-8 / ALPHA:.0e})'
    )


@pytest.mark.base
def test_whole_preconditioner_at_reduced_precision():
    """Both together, which is the configuration worth having: the whole preconditioner is cheap."""
    full, niter_full = run()
    reduced, niter_reduced = run(solve_precision='complex64', transform_precision='complex64')

    assert reduced < FLOOR_FULL, f'the reduced-precision preconditioner capped the residual at {reduced:.3e}'
    assert niter_reduced <= niter_full + 2, (
        f'reduced precision cost {niter_reduced - niter_full} extra iterations, which is more than '
        'a perturbed preconditioner should -- it is supposed to cost rate, and barely that'
    )


@pytest.mark.base
def test_reduced_precision_state_does_cap_accuracy():
    """
    The control. Reduce a precision *outside* the preconditioner and the floor must move.

    Without this the three tests above are indistinguishable from a harness that stopped varying
    anything. Storing the solution at complex64 is the one thing the increment formulation cannot
    protect against, because that error is proportional to ``|u|`` rather than to ``|delta|``.
    """
    import numpy as np
    from pySDC.projects.DeltaSDC.paradiag import FULL, controller_ParaDiag_reduced_transform

    class LowPrecisionState(controller_ParaDiag_reduced_transform):
        def update_solution(self, local_MS_running):
            super().update_solution(local_MS_running)
            for S in local_MS_running:
                for m in range(S.levels[0].sweep.coll.num_nodes):
                    u = S.levels[0].u[m + 1]
                    u[:] = np.asarray(u).astype('complex64').astype(FULL)

    full, _ = run()
    capped, _ = run(controller_class=LowPrecisionState)

    assert capped > 1e4 * full, (
        f'storing the solution at complex64 reached {capped:.3e}, barely worse than the {full:.3e} '
        'of full precision -- this control is supposed to fail, so the measurement is not looking '
        'at what it claims to'
    )
