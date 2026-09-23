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


def as_numpy(a):
    """A pySDC datatype as a plain NumPy array, whichever array library it lives in.

    Two things make this less trivial than `np.asarray`: a CuPy array refuses to be converted
    implicitly and hands over a NumPy copy through `.get()` instead, and the wrapper has to come
    off either way, because `mesh.__abs__` is a norm rather than an elementwise absolute value.
    """
    return np.asarray(a.get() if hasattr(a, 'get') else a)


def total_rhs(f):
    """Sum an rhs over whatever splitting the variant uses, so all variants become comparable."""
    for parts in (('impl', 'expl'), ('comp1', 'comp2')):
        if hasattr(f, parts[0]):
            return sum(as_numpy(getattr(f, part)) for part in parts)
    return as_numpy(f)


def problem_class(cls_name):
    """One Allen-Cahn class by name, with the extra arguments its constructor wants."""
    if cls_name in FFT_VARIANTS:
        import pySDC.implementations.problem_classes.AllenCahn_2D_FFT as FFT

        return getattr(FFT, cls_name), {'L': L}
    if cls_name == 'allencahn_imex':
        from pySDC.implementations.problem_classes.AllenCahn_MPIFFT import allencahn_imex

        return allencahn_imex, {'L': L, 'spectral': False, 'dw': 0.0}

    import pySDC.implementations.problem_classes.AllenCahn_2D_FD as FD

    return getattr(FD, cls_name), {}


def build(cls_name, **kwargs):
    """One Allen-Cahn problem by name, at the shared parameters."""
    cls, extra = problem_class(cls_name)

    return cls(nvars=NVARS, eps=EPS, radius=RADIUS, **extra, **kwargs)


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
@pytest.mark.parametrize('nu', [2, 4])
def test_nu_still_sets_the_exponent(cls_name, nu):
    """nu survived the change of variables: the term is the +-1 one written in 2u - 1, halved."""
    P = build(cls_name, nu=nu)
    u = np.linspace(-0.25, 1.25, 41)

    v = 2.0 * u - 1.0
    expected = 0.5 / EPS**2 * v * (1.0 - v**nu)
    assert abs(P.reaction(u) - expected).max() < 1e-12, f'nu={nu} is not in the reaction term'

    # and the wells stay where they are for any even nu
    assert abs(P.reaction(np.array([0.0, 0.5, 1.0]))).max() < 1e-12, f'nu={nu} moved the wells'


@pytest.mark.base
@pytest.mark.parametrize('nu', [2, 4])
def test_the_jacobians_follow_nu(nu):
    """The Newton solves would silently mis-converge if the derivative ignored nu."""
    P = build('allencahn_fullyimplicit', nu=nu)
    u = np.linspace(-0.25, 1.25, 41)
    h = 1e-6

    for which in ('reaction', 'reaction_cubic'):
        f, df = getattr(P, which), getattr(P, which + '_prime')
        numerical = (f(u + h) - f(u - h)) / (2 * h)
        error = abs(df(u) - numerical).max() / abs(numerical).max()
        assert error < 1e-7, f'{which}_prime ignores nu={nu}: relative error {error:.2e}'


@pytest.mark.base
@pytest.mark.parametrize('nu', [2, 4])
def test_the_split_halves_follow_nu(nu):
    """The _v2 variants solve the halves separately, so they have to sum back for any nu."""
    P = build('allencahn_semiimplicit_v2', nu=nu)
    u = np.linspace(-0.25, 1.25, 41)

    assert abs(P.reaction_cubic(u) + P.reaction_linear(u) - P.reaction(u)).max() < 1e-12


@pytest.mark.base
@pytest.mark.parametrize('which', ['reaction', 'reaction_cubic'])
def test_the_jacobians_are_the_derivatives_they_claim(which):
    """The Newton solves stand or fall on these, and nothing else checks them."""
    P = build('allencahn_fullyimplicit')
    f = getattr(P, which)
    df = getattr(P, which + '_prime')

    u = np.linspace(-0.25, 1.25, 41)
    h = 1e-6
    numerical = (f(u + h) - f(u - h)) / (2 * h)

    error = abs(df(u) - numerical).max() / abs(numerical).max()
    assert error < 1e-8, f'{which}_prime is not the derivative of {which}: relative error {error:.2e}'


# --------------------------------------------------------------------------------------------
# the classes against each other
# --------------------------------------------------------------------------------------------


@pytest.mark.base
@pytest.mark.parametrize('cls_name', FD_VARIANTS[1:] + FFT_VARIANTS)
def test_one_reaction_term_everywhere(cls_name):
    """The term is spelled out once per file, so pin the classes to each other against drift.

    Checking the roots is not enough: a wrong coefficient keeps them where they are.
    """
    u = np.linspace(-0.25, 1.25, 41)
    reference = build('allencahn_fullyimplicit').reaction(u)

    assert abs(build(cls_name).reaction(u) - reference).max() == 0.0, f'{cls_name} reacts differently'


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


@pytest.mark.base
@pytest.mark.parametrize('cls_name', FD_VARIANTS + FFT_VARIANTS)
def test_the_grid_is_centred(cls_name):
    """Every class discretizes [-L/2, L/2), so the blob sits at the origin and diagnostics line up."""
    x = np.asarray(build(cls_name).xvalues)
    dx = L / NVARS[0]

    assert x.min() == pytest.approx(-L / 2), f'grid starts at {x.min()}, not -L/2'
    assert x.max() == pytest.approx(L / 2 - dx), f'grid ends at {x.max()}, not L/2 - dx'


@pytest.mark.mpi4py
def test_MPIFFT_shares_that_grid():
    """It used to run [0, L) while its own docstring claimed otherwise."""
    x = np.asarray(build('allencahn_imex').X[0])
    dx = L / NVARS[0]

    assert x.min() == pytest.approx(-L / 2), f'grid starts at {x.min()}, not -L/2'
    assert x.max() == pytest.approx(L / 2 - dx), f'grid ends at {x.max()}, not L/2 - dx'


@pytest.mark.mpi4py
@pytest.mark.parametrize('domain', [1.0, 2.0])
def test_the_blob_sits_at_the_centre_for_any_L(domain):
    """The circle used to be pinned at (0.5, 0.5) whatever L was, so it was off-centre unless L=1."""
    from pySDC.implementations.problem_classes.AllenCahn_MPIFFT import allencahn_imex

    n = 64
    P = allencahn_imex(nvars=(n, n), eps=EPS, radius=RADIUS, L=domain, spectral=False, dw=0.0)
    u = np.asarray(P.u_exact(0.0))

    assert u[n // 2, n // 2] == u.max(), f'at L={domain} the blob peaks off centre'


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


@pytest.mark.base
@pytest.mark.parametrize('cls_name', FD_VARIANTS + FFT_VARIANTS)
def test_the_work_counters_count(cls_name):
    """Every class counts its right-hand sides, and every solver its iterations."""
    P = build(cls_name)
    u = P.u_exact(0.0)

    assert P.work_counters['rhs'].niter == 0, 'the rhs counter did not start at zero'
    P.eval_f(u, 0.0)
    assert P.work_counters['rhs'].niter == 1, 'evaluating the rhs did not count'

    # the FFT classes invert the Laplacian in one shot and have nothing to iterate
    if not {'newton', 'linear'} & set(P.work_counters):
        return

    # the FD ones do. They still track how many times each solver was *called*, which the work
    # counters do not, so use that: whenever a solver ran, its iterations have to have been
    # counted. One of them is a closed-form solve and calls neither, which is why this is tied
    # to the call counters rather than simply demanding that every solver iterate.
    def state():
        return (
            P.newton_ncalls,
            P.lin_ncalls,
            P.work_counters['newton'].niter,
            P.work_counters['linear'].niter,
        )

    for solver in ('solve_system', 'solve_system_1', 'solve_system_2'):
        if not hasattr(P, solver):
            continue
        n0, l0, wn0, wl0 = state()
        getattr(P, solver)(u, 1e-4, u, 0.0)
        n1, l1, wn1, wl1 = state()

        assert n1 == n0 or wn1 > wn0, f'{solver} ran a Newton solve without counting its iterations'
        assert l1 == l0 or wl1 > wl0, f'{solver} ran a linear solve without counting its iterations'


@pytest.mark.cupy
@pytest.mark.parametrize('cls_name', FD_VARIANTS + FFT_VARIANTS)
def test_the_GPU_path_matches_the_CPU_one(cls_name):
    """These used to be separate files that drifted apart; now useGPU is the only difference."""
    cls, extra = problem_class(cls_name)
    params = dict(nvars=NVARS, eps=EPS, radius=RADIUS, **extra)

    # setup_GPU switches the class rather than the instance, so run the GPU side on a throwaway
    # subclass and leave the shared class on the CPU for the rest of the suite
    cpu = cls(**params)
    on_gpu = type(f'{cls_name}_on_GPU', (cls,), {})
    gpu = on_gpu(useGPU=True, **params)

    # a multilevel run builds one problem per level, so setup_GPU has to survive a second call
    on_gpu(useGPU=True, **params)

    u_cpu, u_gpu = as_numpy(cpu.u_exact(0.0)), as_numpy(gpu.u_exact(0.0))
    assert abs(u_cpu - u_gpu).max() < 1e-12, 'the two modes start from different fields'

    f_cpu = total_rhs(cpu.eval_f(cpu.u_exact(0.0), 0.0))
    f_gpu = total_rhs(gpu.eval_f(gpu.u_exact(0.0), 0.0))
    assert abs(f_cpu - f_gpu).max() < 1e-10 * max(abs(f_cpu).max(), 1.0), 'the two modes disagree on the rhs'


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
