"""
The DeltaSDC precision ladder on a real GPU.

The CPU tests establish what reduced precision costs; these check that the same holds once the
arrays live on the device, and that the reduced precision is genuine there -- CuPy, like NumPy,
upcasts silently, and an upcast would pass every accuracy check. Small grids: the timing is
``run_gpu.py``'s business, not a test's.
"""

import functools

import pytest

N = 64


def _run(label):
    from pySDC.projects.DeltaSDC.run_gpu import configurations, run

    config = {c[0]: c[1:] for c in configurations()}[label]
    return run(N, *config, use_gpu=True, dt=1e-2, nsteps=1)


@pytest.mark.cupy
def test_ladder_matches_fp64():
    """Every delta-form row reaches the fp64 answer with the fp64 iteration count."""
    reference, _, sdc_iter, _, _ = _run('SDC, CG 1e-12')
    expected = {
        'deltaSDC, CG 1e-5': (sdc_iter, 1e-12),
        'deltaSDC, fp32 CG 1e-5': (sdc_iter, 1e-12),
        'deltaMLSDC, fp32 fine CG + fp32 coarse': (None, 1e-10),
        'deltaMLSDC, fp32 fine CG + fp16 coarse': (None, 1e-10),
    }
    ml_iter = _run('MLSDC, CG 1e-12')[2]
    for label, (niter, tol) in expected.items():
        uend, _, got, _, _ = _run(label)
        assert abs(uend - reference) < tol, label
        assert got == (niter or ml_iter), f'{label}: {got} iterations'


@pytest.mark.cupy
def test_control_stalls():
    """Stock SDC handed the loose tolerance does not get there: what licenses it is the delta form."""
    reference = _run('SDC, CG 1e-12')[0]
    uend, _, niter, _, _ = _run('CONTROL SDC, CG 1e-5')
    # the converged rows sit ~1e-13 from the reference; restol is 1e-10
    assert niter == 50 and abs(uend - reference) > 1e-10


@pytest.mark.cupy
def test_reduced_precision_is_genuine():
    """The coarse level really holds reduced device arrays, and the fp32 solve really is fp32."""
    import cupy as cp

    for coarse_dtype in ['fp32', 'fp16']:
        *_, controller = _run(f'deltaMLSDC, fp32 fine CG + {coarse_dtype} coarse')
        fine, coarse = controller.MS[0].levels
        expected = cp.float32 if coarse_dtype == 'fp32' else cp.float16
        assert all(isinstance(u, cp.ndarray) and u.dtype == expected for u in coarse.u)
        assert all(u.dtype == cp.float64 for u in fine.u)

    # fingerprint: same system, fp32 vs fp64 solve, must differ by about fp32's epsilon
    prob = fine.prob
    rhs = prob.dtype_u(prob.init)
    rhs[:] = cp.random.default_rng(0).standard_normal(prob.nvars)
    prob.lintol = 1e-12
    single = prob.solve_system(rhs, 0.01, None, 0.0)
    prob.solve_dtype, keep = None, prob.solve_dtype
    double = prob.solve_system(rhs, 0.01, prob.dtype_u(prob.init, val=0.0), 0.0)
    prob.solve_dtype = keep
    assert single.dtype == cp.float64
    assert 1e-9 < float(abs(single - double) / abs(double)) < 1e-5


@pytest.mark.cupy
def test_paradiag_complex64_on_device():
    """The whole ParaDiag preconditioner at complex64 on the GPU: fp64's answer, genuinely single."""
    import cupy as cp

    from pySDC.projects.DeltaSDC.paradiag import heat_paradiag
    from pySDC.projects.DeltaSDC.run_gpu import run_paradiag

    reference, _, niter_full, _ = run_paradiag(N, None, None, True, 1e-2, 8)
    uend, _, niter, residual = run_paradiag(N, 'complex64', 'complex64', True, 1e-2, 8)
    assert isinstance(uend, cp.ndarray) and uend.dtype == cp.complex128
    assert float(abs(uend - reference)) < 1e-12 and residual < 1e-10
    assert niter <= niter_full + 1

    # fingerprint: the complex64 solve must differ from the complex128 one by about its epsilon
    solves = []
    for precision in [None, 'complex64']:
        prob = heat_paradiag(nvars=(N, N), nu=0.1, freq=(2, 2), bc='periodic', solve_precision=precision, useGPU=True)
        rhs = prob.dtype_u(prob.init)
        rhs[:] = cp.random.default_rng(0).standard_normal(prob.nvars)
        solves.append(prob.solve_jacobian(rhs, 0.3 - 0.2j))
    assert prob._eigenvalues.dtype == cp.complex64
    assert 1e-9 < float(abs(solves[1] - solves[0]) / abs(solves[0])) < 1e-5


# The README's two demo tables, as measured on a T4 under NumPy 2. Iteration counts are exact: the GPU
# reproduces the CPU run's, and a change here is a change in behaviour, not noise. Controls are `None`.
HEAT_TABLE = {
    'SDC': 14,
    'fp32-deltaSDC': 14,
    'fp16-deltaSDC': 15,
    'MLSDC': 7,
    'deltaMLSDC': 7,
    'fp32-coarse-solve': 7,
    'fp16-coarse-solve': 7,
    'fp16 coarse level and solve': 7,
    'genuine fp32 coarse level': 7,
    'genuine fp16 coarse level': 11,
    'all three: fp32 solve, fp16 coarse, cascade': 8,
    'full ladder: fp32 fine, fp16 coarse': 10,
    'CONTROL fp16 coarse, stock ML': None,
    'CONTROL fp16 solve, unnormalised': None,
    'CONTROL fp32 fine level': None,
}

ALLEN_CAHN_SWEEPS = {
    'SDC': 18,
    'deltaSDC': 18,
    'fp32-deltaSDC': 18,
    'MLSDC': 20,
    'deltaMLSDC': 20,
    'fp32-deltaMLSDC': 20,
    'fp16-coarse-deltaMLSDC': 20,
    'cascade fp16>fp32>fp64 fine': 20,
    'CONTROL fp16 coarse, stock ML': 60,
    'CONTROL fp32 fine level': 60,
}


@functools.cache
def _heat_row(label):
    from pySDC.projects.DeltaSDC.run_demo import run_heat_configuration

    return run_heat_configuration(label, useGPU=True)


@functools.cache
def _allen_cahn_row(label):
    from pySDC.projects.DeltaSDC.run_demo import run_configuration

    return run_configuration(label, useGPU=True)


# One test per row, so that xdist spreads a table over the GPUs rather than one worker running all of
# it; the reference rows are cached per worker, so each is computed at most once on each.


@pytest.mark.cupy
@pytest.mark.parametrize('label', list(HEAT_TABLE))
def test_readme_heat_table_on_gpu(label):
    """The linear precision ladder, row by row, on the device."""
    uend, hit, floor = _heat_row(label)
    expected = HEAT_TABLE[label]
    assert hit == expected, f'{label}: {hit} iterations, the README has {expected}'
    if expected is None:
        assert floor > 1e-6, f'{label} is a control and must stall, but reached {floor:.1e}'
    else:
        assert float(abs(uend - _heat_row('SDC')[0])) < 1e-14


@pytest.mark.cupy
@pytest.mark.parametrize('label', list(ALLEN_CAHN_SWEEPS))
def test_readme_allen_cahn_table_on_gpu(label):
    """The nonlinear table, row by row: Newton-CG correction solves at reduced precision, on the device."""
    from pySDC.projects.DeltaSDC.run_demo import configurations

    result = _allen_cahn_row(label)
    expected = ALLEN_CAHN_SWEEPS[label]
    assert result['niter'] == expected, f"{label}: {result['niter']} sweeps, the README has {expected}"
    multilevel = next(config[-1] for config in configurations() if config[0] == label).get('multilevel')
    diff = float(abs(result['uend'] - _allen_cahn_row('MLSDC' if multilevel else 'SDC')['uend']))
    assert (diff > 1e-8) if label.startswith('CONTROL') else (diff < 1e-12), f'{label}: {diff:.1e} off'


@pytest.mark.base
def test_readme_tables_are_complete():
    """Every row of both tables is checked, and nothing checked has left the tables."""
    from pySDC.projects.DeltaSDC.run_demo import configurations, heat_configurations

    assert [config[0] for config in heat_configurations()] == list(HEAT_TABLE)
    assert [config[0] for config in configurations()] == list(ALLEN_CAHN_SWEEPS)


@pytest.mark.cupy
def test_half_precision_fft_solve_is_genuine():
    """cuFFT's complex32 transform: off by half precision's epsilon, not by single's, and not broken."""
    import cupy as cp

    from pySDC.projects.DeltaSDC.problems import heat_solve_dtype

    solves = {}
    for precision in [None, 'float16']:
        prob = heat_solve_dtype(
            nvars=(256, 256), nu=0.1, bc='periodic', solver_type='FFT', solve_dtype=precision, useGPU=True
        )
        rhs = prob.dtype_u(prob.init)
        rhs[:] = cp.random.default_rng(0).standard_normal(prob.nvars) * 1e-9
        solves[precision] = prob.solve_system(rhs, 1e-3, None, 0.0)
    error = float(abs(solves['float16'] - solves[None]) / abs(solves[None]))
    # single precision would be ~1e-7; a broken transform or a flushed rhs would be ~1
    assert 1e-5 < error < 1e-2, f'half-precision solve off by {error:.1e}'
    assert solves['float16'].dtype == cp.float64


@pytest.mark.cupy
def test_half_precision_fft_follows_the_delivered_accuracy_spec():
    """
    The README's specification, with genuine half-precision arithmetic: MLSDC's fine solve is the
    demanding one, its coarse solve the forgiving one, and every row still reaches the answer.
    """
    from pySDC.projects.DeltaSDC.run_gpu import fft_configurations, run

    rows = {label: run(256, *config, use_gpu=True, dt=1e-2, nsteps=1) for label, *config in fft_configurations()}
    iters = {label: row[2] for label, row in rows.items()}
    for label, row in rows.items():
        peer = rows['MLSDC, FFT' if 'ML' in label else 'SDC, FFT'][0]
        assert float(abs(row[0] - peer)) < 1e-10, f'{label} did not reach its fp64 peer'

    assert iters['deltaSDC, fp32 FFT'] == iters['SDC, FFT'], 'single precision should be free'
    assert iters['deltaSDC, fp16 FFT'] <= iters['SDC, FFT'] + 2
    assert iters['MLSDC, FFT'] <= iters['deltaMLSDC, fp16 coarse FFT'] <= iters['MLSDC, FFT'] + 1
    assert (
        iters['deltaMLSDC, fp16 fine FFT'] > iters['deltaMLSDC, fp16 coarse FFT']
    ), f'the fine solve should need more accuracy than the coarse one: {iters}'
