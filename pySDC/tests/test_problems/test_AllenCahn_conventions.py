r"""
Consistency tests across the Allen-Cahn implementations (see issue #434).

pySDC used to spell Allen-Cahn two ways. The FD and FFT classes solved

.. math:: u_t = \Delta u + \frac{1}{\varepsilon^2} u (1 - u^\nu),\qquad \nu = 2,

with wells at :math:`u = \pm 1`, while the MPIFFT classes solved

.. math:: u_t = \Delta u - \frac{2}{\varepsilon^2} u (1 - u)(1 - 2u),

with wells at :math:`u = 0` and :math:`u = 1`. Substituting :math:`u = (1 + v)/2` turns the second
into the first at the same :math:`\varepsilon` and the same time scale, so they were one problem
in two variables -- which is why monitoring hooks, initial conditions and diagnostics kept
disagreeing with each other.

Everything now uses the :math:`0` to :math:`1` form, and these tests hold it there. Most of them
are absolute: they check against the analytic Allen-Cahn problem rather than against a sibling
class, because a check that only compares classes to each other passes happily when they are
wrong together.

The Laplacian is the one piece that legitimately differs between the classes -- second-order FD
against spectral -- so the cross-discretization checks compare only the parts that are
discretization-free.
"""

import numpy as np
import pytest

NVARS = (64, 64)
EPS = 0.04
RADIUS = 0.25
L = 1.0

FD_VARIANTS = [
    'allencahn_fullyimplicit',
    'allencahn_semiimplicit',
    'allencahn_semiimplicit_v2',
    'allencahn_multiimplicit',
    'allencahn_multiimplicit_v2',
]
FFT_VARIANTS = ['allencahn2d_imex', 'allencahn2d_imex_stab']


def total_rhs(f):
    """Sum an rhs over whatever splitting the variant uses, so all variants become comparable."""
    for parts in (('impl', 'expl'), ('comp1', 'comp2')):
        if hasattr(f, parts[0]):
            return sum(np.asarray(getattr(f, part)) for part in parts)
    return np.asarray(f)


def build(cls_name, **kwargs):
    """One Allen-Cahn problem by name, at the shared parameters."""
    params = dict(nvars=NVARS, eps=EPS, radius=RADIUS, **kwargs)

    if cls_name in FFT_VARIANTS:
        import pySDC.implementations.problem_classes.AllenCahn_2D_FFT as FFT

        return getattr(FFT, cls_name)(L=L, **params)
    if cls_name == 'allencahn_imex':
        from pySDC.implementations.problem_classes.AllenCahn_MPIFFT import allencahn_imex

        return allencahn_imex(L=L, spectral=False, dw=0.0, **params)

    import pySDC.implementations.problem_classes.AllenCahn_2D_FD as FD

    return getattr(FD, cls_name)(**params)


# --------------------------------------------------------------------------------------------
# absolute checks: is this the 0..1 Allen-Cahn problem at all?
# --------------------------------------------------------------------------------------------


@pytest.mark.base
@pytest.mark.parametrize('cls_name', FD_VARIANTS + FFT_VARIANTS)
def test_the_wells_are_at_zero_and_one(cls_name):
    """The defining property of this form: the reaction vanishes at 0, 1/2 and 1, and nowhere else."""
    P = build(cls_name)

    roots = np.array([0.0, 0.5, 1.0])
    assert abs(P.reaction(roots)).max() == 0.0, 'the reaction does not vanish at the wells'

    # and it pushes away from 1/2, towards whichever well is nearer
    assert P.reaction(np.array([0.25])) < 0, 'the low side does not fall towards 0'
    assert P.reaction(np.array([0.75])) > 0, 'the high side does not climb towards 1'

    # the v2 variants split the reaction in two and solve the halves separately, so the halves
    # have to add back up to it exactly
    if hasattr(P, 'reaction_cubic'):
        u = np.linspace(-0.5, 1.5, 21)
        assert (
            abs(P.reaction_cubic(u) + P.reaction_linear(u) - P.reaction(u)).max() < 1e-12
        ), 'the split halves do not sum to the reaction term'


@pytest.mark.base
@pytest.mark.parametrize('cls_name', FD_VARIANTS + FFT_VARIANTS)
def test_the_initial_blob_runs_from_zero_to_one(cls_name):
    """A saturated circle, so the field has to span the two wells and stay inside them."""
    u = np.asarray(build(cls_name).u_exact(0.0))

    # a blob only a few epsilon across does not saturate all the way, but it may not overshoot
    # either, and under the old +-1 convention the minimum would have been near -1
    assert 0.0 <= u.min() < 1e-3, f'low phase sits at {u.min()}, not 0'
    assert 1.0 - 1e-3 < u.max() <= 1.0, f'high phase sits at {u.max()}, not 1'


@pytest.mark.base
@pytest.mark.parametrize('cls_name', FD_VARIANTS + FFT_VARIANTS)
def test_nu_is_rejected(cls_name):
    """nu belonged to the +-1 polynomial and has no analogue here, so it must not be ignored."""
    from pySDC.core.errors import ProblemError

    with pytest.raises(ProblemError, match='nu'):
        build(cls_name, nu=3)


# --------------------------------------------------------------------------------------------
# the classes against each other
# --------------------------------------------------------------------------------------------


@pytest.mark.base
@pytest.mark.parametrize('cls_name', FD_VARIANTS[1:] + FFT_VARIANTS)
def test_one_initial_condition_everywhere(cls_name):
    """Different discretizations, but the same analytic blob sampled on the same grid."""
    reference = np.asarray(build('allencahn_fullyimplicit').u_exact(0.0))
    u = np.asarray(build(cls_name).u_exact(0.0))

    assert abs(u - reference).max() == 0.0, f'{cls_name} starts from a different field'


@pytest.mark.base
@pytest.mark.parametrize('cls_name', FD_VARIANTS[1:])
def test_FD_variants_agree_on_the_rhs(cls_name):
    """The FD variants only differ in how they split the rhs, never in what it sums to."""
    reference = total_rhs(build('allencahn_fullyimplicit').eval_f(build('allencahn_fullyimplicit').u_exact(0.0), 0.0))
    P = build(cls_name)
    f = total_rhs(P.eval_f(P.u_exact(0.0), 0.0))

    assert abs(f - reference).max() < 1e-12, f'{cls_name} sums to a different rhs'


@pytest.mark.mpi4py
def test_FFT_and_MPIFFT_are_now_the_same_problem():
    """Same equation, same convention, same spectral discretization: nothing may differ."""
    A, B = build('allencahn2d_imex'), build('allencahn_imex')
    uA, uB = np.asarray(A.u_exact(0.0)), np.asarray(B.u_exact(0.0))

    assert abs(uA - uB).max() == 0.0, 'initial conditions differ'

    fA, fB = A.eval_f(uA, 0.0), B.eval_f(uB, 0.0)
    for part in ('impl', 'expl'):
        a, b = np.asarray(getattr(fA, part)), np.asarray(getattr(fB, part))
        assert abs(a - b).max() < 1e-10 * max(abs(b).max(), 1.0), f'f.{part} differs'


@pytest.mark.mpi4py
def test_FD_reaction_matches_MPIFFT():
    """The reaction term is pointwise, so it agrees exactly even across discretizations."""
    A, B = build('allencahn_semiimplicit'), build('allencahn_imex')
    a = np.asarray(A.eval_f(A.u_exact(0.0), 0.0).expl)
    b = np.asarray(B.eval_f(B.u_exact(0.0), 0.0).expl)

    assert abs(a - b).max() < 1e-10 * abs(b).max(), 'the FD reaction term disagrees'


@pytest.mark.mpi4py
def test_time_stepping_agrees():
    """End to end: the two spectral classes must now integrate to the same solution."""
    from pySDC.implementations.problem_classes.AllenCahn_2D_FFT import allencahn2d_imex
    from pySDC.implementations.problem_classes.AllenCahn_MPIFFT import allencahn_imex

    dt, nsteps = 1e-4, 4
    uA = _run(allencahn2d_imex, {'nvars': NVARS, 'eps': EPS, 'radius': RADIUS, 'L': L}, dt, nsteps)
    uB = _run(
        allencahn_imex, {'nvars': NVARS, 'eps': EPS, 'radius': RADIUS, 'L': L, 'spectral': False, 'dw': 0.0}, dt, nsteps
    )

    error = abs(uA - uB).max()
    assert error < 1e-10, f'the two classes drift apart by {error:.3e} after {nsteps} steps'


def _run(problem_class, problem_params, dt, nsteps, hook=None):
    from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
    from pySDC.implementations.sweeper_classes.imex_1st_order import imex_1st_order

    description = {
        'problem_class': problem_class,
        'problem_params': problem_params,
        'sweeper_class': imex_1st_order,
        'sweeper_params': {'quad_type': 'RADAU-RIGHT', 'num_nodes': 3, 'QI': 'LU'},
        'level_params': {'restol': -1, 'dt': dt},
        'step_params': {'maxiter': 6},
    }
    controller = controller_nonMPI(1, {'logger_level': 30, 'hook_class': hook or []}, description)
    prob = controller.MS[0].levels[0].prob
    uend, stats = controller.run(u0=prob.u_exact(0.0), t0=0.0, Tend=nsteps * dt)

    return stats if hook else np.asarray(uend)


# --------------------------------------------------------------------------------------------
# the monitoring hook
# --------------------------------------------------------------------------------------------


def monitor_output(problem_class, problem_params, dt=5e-4, nsteps=8):
    from pySDC.helpers.stats_helper import get_sorted
    from pySDC.implementations.hooks.AllenCahn_monitor import AllenCahnMonitor

    stats = _run(problem_class, problem_params, dt, nsteps, hook=AllenCahnMonitor)

    return {
        key: np.array([value for _, value in get_sorted(stats, type=key, sortby='time')])
        for key in ('computed_radius', 'exact_radius', 'computed_volume', 'exact_volume')
    }


@pytest.mark.base
@pytest.mark.parametrize('n', [32, 64, 128])
def test_monitor_converges_on_the_true_radius(n):
    """Absolute, and the reason for counting cells: the estimator has to refine away its own bias."""
    from pySDC.implementations.problem_classes.AllenCahn_2D_FFT import allencahn2d_imex

    out = monitor_output(allencahn2d_imex, {'nvars': (n, n), 'eps': EPS, 'radius': RADIUS, 'L': L}, nsteps=1)
    error = abs(out['computed_radius'][0] - RADIUS)

    # the bias is O(dx); allow a couple of cells of slack but demand it shrink with the mesh
    assert error < 3.0 * L / n, f'at n={n} the measured radius is off by {error:.2e}'


@pytest.mark.base
def test_monitor_tracks_the_shrinking_circle():
    """Over a run the measured radius must follow the mean curvature flow law, not merely start right."""
    from pySDC.implementations.problem_classes.AllenCahn_2D_FFT import allencahn2d_imex

    out = monitor_output(allencahn2d_imex, {'nvars': (128, 128), 'eps': EPS, 'radius': RADIUS, 'L': L})

    assert np.all(np.diff(out['computed_radius']) <= 0), 'the blob is not shrinking'
    assert abs(out['computed_radius'] - out['exact_radius']).max() < 5e-3, 'it shrinks at the wrong rate'


@pytest.mark.mpi4py
def test_monitor_agrees_across_discretizations():
    """FD, FFT and MPIFFT are one problem now, so the monitor must report one blob."""
    from pySDC.implementations.problem_classes.AllenCahn_2D_FFT import allencahn2d_imex
    from pySDC.implementations.problem_classes.AllenCahn_MPIFFT import allencahn_imex

    a = monitor_output(allencahn2d_imex, {'nvars': NVARS, 'eps': EPS, 'radius': RADIUS, 'L': L})
    b = monitor_output(
        allencahn_imex, {'nvars': NVARS, 'eps': EPS, 'radius': RADIUS, 'L': L, 'spectral': False, 'dw': 0.0}
    )

    assert abs(a['computed_radius'] - b['computed_radius']).max() == 0.0, 'the two classes measure different blobs'


@pytest.mark.mpi4py
def test_monitor_in_3d():
    """The 3D branch has no other coverage, and its shrinking law carries a different coefficient."""
    from pySDC.implementations.problem_classes.AllenCahn_MPIFFT import allencahn_imex

    dt, nsteps = 2e-4, 4
    out = monitor_output(
        allencahn_imex,
        {'nvars': (32, 32, 32), 'eps': EPS, 'radius': RADIUS, 'L': L, 'spectral': False, 'dw': 0.0},
        dt=dt,
        nsteps=nsteps,
    )

    ball = 4.0 / 3.0 * np.pi * out['computed_radius'] ** 3
    assert abs(ball - out['computed_volume']).max() < 1e-12, 'radius and volume disagree in 3D'

    # mean curvature flow in d dimensions: R^2 = R0^2 - 2 (d - 1) t, so 4t rather than 2t here
    t = np.arange(nsteps + 1) * dt
    assert abs(out['exact_radius'] - np.sqrt(RADIUS**2 - 4.0 * t)).max() < 1e-14, 'wrong shrinking law in 3D'
