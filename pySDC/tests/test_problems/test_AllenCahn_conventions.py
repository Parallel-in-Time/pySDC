r"""
Consistency tests across the Allen-Cahn implementations (see issue #434).

pySDC spells Allen-Cahn two ways. The FD/FFT classes solve

.. math:: u_t = \Delta u + \frac{1}{\varepsilon^2} u (1 - u^\nu),\qquad \nu = 2,

with wells at :math:`u = \pm 1`, while the MPIFFT classes solve

.. math:: u_t = \Delta u - \frac{2}{\varepsilon^2} u (1 - u)(1 - 2u) - 6 d_w u (1 - u),

with wells at :math:`u = 0, 1`. Substituting :math:`u = (1 + v) / 2` into the second turns it into
the first, with the same :math:`\varepsilon` and the same time scale, so for :math:`d_w = 0` these
are one problem under an affine change of variables:

.. math:: u_\mathrm{MPIFFT} = \frac{1 + u_\mathrm{FD/FFT}}{2},
          \qquad f_\mathrm{MPIFFT} = \frac{f_\mathrm{FD/FFT}}{2}.

These tests pin that identity down. They exist so that the convention can be unified without
silently changing the physics: whichever spelling survives, the mapped fields must still agree.

The Laplacian is the one piece that legitimately differs -- second-order FD against spectral -- so
the cross-discretization checks compare only the parts that are discretization-free.
"""

import numpy as np
import pytest

NVARS = (64, 64)
EPS = 0.04
RADIUS = 0.25
L = 1.0


def total_rhs(f):
    """Sum an rhs over whatever splitting the variant uses, so all variants become comparable."""
    for parts in (('impl', 'expl'), ('comp1', 'comp2')):
        if hasattr(f, parts[0]):
            return sum(np.asarray(getattr(f, part)) for part in parts)
    return np.asarray(f)


def family_A(cls_name):
    """One of the +-1 classes, built at the shared parameters."""
    if cls_name == 'allencahn2d_imex':
        from pySDC.implementations.problem_classes.AllenCahn_2D_FFT import allencahn2d_imex

        return allencahn2d_imex(nvars=NVARS, eps=EPS, radius=RADIUS, nu=2, L=L)

    import pySDC.implementations.problem_classes.AllenCahn_2D_FD as FD

    return getattr(FD, cls_name)(nvars=NVARS, eps=EPS, radius=RADIUS, nu=2)


def family_B():
    """The 0..1 class, at the same parameters and with the driving force switched off."""
    from pySDC.implementations.problem_classes.AllenCahn_MPIFFT import allencahn_imex

    return allencahn_imex(nvars=NVARS, eps=EPS, radius=RADIUS, L=L, spectral=False, dw=0.0)


FD_VARIANTS = [
    'allencahn_fullyimplicit',
    'allencahn_semiimplicit',
    'allencahn_semiimplicit_v2',
    'allencahn_multiimplicit',
]


@pytest.mark.base
@pytest.mark.parametrize('cls_name', FD_VARIANTS + ['allencahn2d_imex'])
def test_family_A_shares_one_initial_condition(cls_name):
    """The +-1 classes discretize differently but must start from the very same field."""
    reference = family_A('allencahn_fullyimplicit')
    P = family_A(cls_name)

    assert (
        abs(np.asarray(P.u_exact(0.0)) - np.asarray(reference.u_exact(0.0))).max() == 0.0
    ), f'{cls_name} starts from a different field'

    # the monitoring hook takes no configuring, so each class has to say which wells it has
    assert P.phase_thresh == 0.0, f'{cls_name} does not declare the +-1 convention'


@pytest.mark.base
@pytest.mark.parametrize('cls_name', FD_VARIANTS[1:])
def test_FD_variants_agree_on_the_rhs(cls_name):
    """The FD variants only differ in how they split the rhs, never in what it sums to."""
    reference = total_rhs(_eval_at_t0('allencahn_fullyimplicit'))
    f = total_rhs(_eval_at_t0(cls_name))

    assert abs(f - reference).max() < 1e-12, f'{cls_name} sums to a different rhs'


def _eval_at_t0(cls_name):
    P = family_A(cls_name)
    return P.eval_f(P.u_exact(0.0), 0.0)


@pytest.mark.mpi4py
def test_family_B_declares_the_other_convention():
    """The 0..1 classes say nothing, which is what the hook reads as "the field is the indicator"."""
    assert getattr(family_B(), 'phase_thresh', None) is None, 'the 0..1 class claims the +-1 convention'


@pytest.mark.mpi4py
def test_FFT_and_MPIFFT_are_the_same_problem():
    """Same spectral discretization, so the affine map must hold to roundoff, term by term."""
    A, B = family_A('allencahn2d_imex'), family_B()
    uA, uB = A.u_exact(0.0), B.u_exact(0.0)

    assert abs(np.asarray(uB) - (1 + np.asarray(uA)) / 2).max() == 0.0, 'initial conditions are not affinely related'

    fA, fB = A.eval_f(uA, 0.0), B.eval_f(uB, 0.0)
    for part in ('impl', 'expl'):
        a, b = np.asarray(getattr(fA, part)), np.asarray(getattr(fB, part))
        assert abs(b - a / 2).max() < 1e-10 * max(abs(a).max(), 1.0), f'f.{part} breaks the map'


@pytest.mark.mpi4py
def test_FD_nonlinearity_matches_MPIFFT():
    """The reaction term is pointwise, so it maps exactly even across discretizations."""
    A, B = family_A('allencahn_semiimplicit'), family_B()
    fA = A.eval_f(A.u_exact(0.0), 0.0)
    fB = B.eval_f(B.u_exact(0.0), 0.0)

    a, b = np.asarray(fA.expl), np.asarray(fB.expl)
    assert abs(b - a / 2).max() < 1e-10 * abs(a).max(), 'the FD reaction term breaks the map'


@pytest.mark.mpi4py
def test_the_map_survives_time_stepping():
    """End to end: integrate both spellings and the solutions still differ only by the map."""
    from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
    from pySDC.implementations.problem_classes.AllenCahn_2D_FFT import allencahn2d_imex
    from pySDC.implementations.problem_classes.AllenCahn_MPIFFT import allencahn_imex
    from pySDC.implementations.sweeper_classes.imex_1st_order import imex_1st_order

    dt, nsteps = 1e-4, 4

    def run(problem_class, problem_params):
        description = {
            'problem_class': problem_class,
            'problem_params': problem_params,
            'sweeper_class': imex_1st_order,
            'sweeper_params': {'quad_type': 'RADAU-RIGHT', 'num_nodes': 3, 'QI': 'LU'},
            'level_params': {'restol': -1, 'dt': dt},
            'step_params': {'maxiter': 8},
        }
        controller = controller_nonMPI(num_procs=1, controller_params={'logger_level': 30}, description=description)
        prob = controller.MS[0].levels[0].prob
        uend, _ = controller.run(u0=prob.u_exact(0.0), t0=0.0, Tend=nsteps * dt)
        return np.asarray(uend)

    uA = run(allencahn2d_imex, {'nvars': NVARS, 'eps': EPS, 'radius': RADIUS, 'nu': 2, 'L': L})
    uB = run(allencahn_imex, {'nvars': NVARS, 'eps': EPS, 'radius': RADIUS, 'L': L, 'spectral': False, 'dw': 0.0})

    error = abs(uB - (1 + uA) / 2).max()
    assert error < 1e-10, f'the two spellings drift apart by {error:.3e} after {nsteps} steps'


def run_with_monitor(problem_class, problem_params, hook, dt=5e-4, nsteps=8):
    from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
    from pySDC.implementations.sweeper_classes.imex_1st_order import imex_1st_order
    from pySDC.helpers.stats_helper import get_sorted

    description = {
        'problem_class': problem_class,
        'problem_params': problem_params,
        'sweeper_class': imex_1st_order,
        'sweeper_params': {'quad_type': 'RADAU-RIGHT', 'num_nodes': 3, 'QI': 'LU'},
        'level_params': {'restol': -1, 'dt': dt},
        'step_params': {'maxiter': 6},
    }
    controller = controller_nonMPI(1, {'logger_level': 30, 'hook_class': hook}, description)
    prob = controller.MS[0].levels[0].prob
    _, stats = controller.run(u0=prob.u_exact(0.0), t0=0.0, Tend=nsteps * dt)

    return {
        key: np.array([value for _, value in get_sorted(stats, type=key, sortby='time')])
        for key in ('computed_radius', 'exact_radius', 'computed_volume', 'exact_volume')
    }


@pytest.mark.mpi4py
def test_one_monitor_serves_both_conventions():
    """The monitor takes no configuring: the problem's own phase_thresh decides what it does."""
    from pySDC.implementations.hooks.AllenCahn_monitor import AllenCahnMonitor
    from pySDC.implementations.problem_classes.AllenCahn_2D_FFT import allencahn2d_imex
    from pySDC.implementations.problem_classes.AllenCahn_MPIFFT import allencahn_imex

    counted = run_with_monitor(
        allencahn2d_imex, {'nvars': NVARS, 'eps': EPS, 'radius': RADIUS, 'nu': 2, 'L': L}, AllenCahnMonitor
    )
    summed = run_with_monitor(
        allencahn_imex,
        {'nvars': NVARS, 'eps': EPS, 'radius': RADIUS, 'L': L, 'spectral': False, 'dw': 0.0},
        AllenCahnMonitor,
    )

    # integrating the 0..1 field picks up an O(eps) bias that no mesh refinement removes, so that
    # estimator is calibrated against t = 0 and has to land on the initial radius exactly
    assert summed['computed_radius'][0] == RADIUS, 'the integrating estimator is not calibrated'

    # counting cells above the threshold is consistent instead, so it is left alone -- and is
    # therefore allowed to be off at t = 0, but only by the mesh
    dx = L / NVARS[0]
    assert 0 < abs(counted['computed_radius'][0] - RADIUS) < dx, 'the counting estimator drifted'

    # either way it is the same shrinking circle
    assert abs(summed['computed_radius'] - summed['exact_radius']).max() < 1e-3, 'the blob shrinks wrongly'
    assert abs(counted['computed_radius'] - summed['computed_radius']).max() < 5e-3, 'the conventions disagree'


@pytest.mark.mpi4py
def test_monitor_in_3d():
    """The 3D branch has no other coverage, and its shrinking law carries a different coefficient."""
    from pySDC.implementations.hooks.AllenCahn_monitor import AllenCahnMonitor
    from pySDC.implementations.problem_classes.AllenCahn_MPIFFT import allencahn_imex

    dt, nsteps = 2e-4, 4
    out = run_with_monitor(
        allencahn_imex,
        {'nvars': (32, 32, 32), 'eps': EPS, 'radius': RADIUS, 'L': L, 'spectral': False, 'dw': 0.0},
        AllenCahnMonitor,
        dt=dt,
        nsteps=nsteps,
    )

    # radius and volume have to describe the same ball
    ball = 4.0 / 3.0 * np.pi * out['computed_radius'] ** 3
    assert abs(ball - out['computed_volume']).max() < 1e-12, 'radius and volume disagree in 3D'

    # mean curvature flow in d dimensions: R^2 = R0^2 - 2 (d - 1) t, so 4t rather than 2t here
    t = np.arange(nsteps + 1) * dt
    assert abs(out['exact_radius'] - np.sqrt(RADIUS**2 - 4.0 * t)).max() < 1e-14, 'wrong shrinking law in 3D'


@pytest.mark.mpi4py
def test_interface_width_is_convention_independent():
    """Measured relative to the wells, the interface is the same width in either spelling."""
    from pySDC.implementations.hooks.AllenCahn_monitor import AllenCahnMonitor
    from pySDC.implementations.problem_classes.AllenCahn_2D_FFT import allencahn2d_imex
    from pySDC.implementations.problem_classes.AllenCahn_MPIFFT import allencahn_imex

    def width(problem_class, problem_params):
        prob = problem_class(**problem_params)
        monitor = AllenCahnMonitor()
        monitor.phase_thresh = getattr(prob, 'phase_thresh', None)
        level = type('level', (), {'prob': prob})
        return monitor.get_interface_width(level, np.asarray(prob.u_exact(0.0)))

    counted = width(allencahn2d_imex, {'nvars': NVARS, 'eps': EPS, 'radius': RADIUS, 'nu': 2, 'L': L})
    summed = width(allencahn_imex, {'nvars': NVARS, 'eps': EPS, 'radius': RADIUS, 'L': L, 'spectral': False, 'dw': 0.0})

    assert counted == summed, f'interface width depends on the convention: {counted} vs {summed}'


if __name__ == '__main__':
    test_FFT_and_MPIFFT_are_the_same_problem()
    test_FD_nonlinearity_matches_MPIFFT()
    test_the_map_survives_time_stepping()
    test_one_monitor_serves_both_conventions()
    test_monitor_in_3d()
    test_interface_width_is_convention_independent()
    print('ok')
