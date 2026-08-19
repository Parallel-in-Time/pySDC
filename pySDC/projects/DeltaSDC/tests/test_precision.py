import pytest


@pytest.mark.base
def test_working_dtype_and_eps():
    import numpy as np
    from pySDC.projects.DeltaSDC.precision import working_dtype, working_eps

    assert working_dtype(None) == np.dtype('float64')
    assert working_dtype(np.float32) == np.dtype('float32')
    assert working_dtype('float16') == np.dtype('float16')

    assert working_eps(None) == pytest.approx(np.finfo(np.float64).eps)
    assert working_eps(np.float32) == pytest.approx(np.finfo(np.float32).eps)


@pytest.mark.base
@pytest.mark.parametrize('dtype', [None, 'float64', 'float32', 'float16'])
def test_tol_floor_scales_with_eps(dtype):
    import numpy as np
    from pySDC.projects.DeltaSDC.precision import tol_floor, working_eps

    assert tol_floor(dtype) == pytest.approx(100.0 * working_eps(dtype))
    assert tol_floor(dtype, safety=10.0) == pytest.approx(10.0 * working_eps(dtype))
    assert tol_floor(np.float32) > tol_floor(np.float64)


@pytest.mark.base
def test_clamp_tolerance():
    import numpy as np
    from pySDC.projects.DeltaSDC.precision import clamp_tolerance, tol_floor

    floor32 = tol_floor(np.float32)

    value, clamped = clamp_tolerance(None, np.float32)
    assert value == pytest.approx(floor32) and clamped is False

    value, clamped = clamp_tolerance(1e-2, np.float32)
    assert value == pytest.approx(1e-2) and clamped is False

    value, clamped = clamp_tolerance(1e-12, np.float32)
    assert value == pytest.approx(floor32) and clamped is True

    value, clamped = clamp_tolerance(1e-12, None)
    assert value == pytest.approx(1e-12) and clamped is False


@pytest.mark.base
def test_precision_aware_mixin_reports():
    import numpy as np
    from pySDC.projects.DeltaSDC.precision import PrecisionAwareTolerances, tol_floor

    class Dummy(PrecisionAwareTolerances):
        pass

    obj = Dummy()
    report = obj.setup_precision_tolerances(np.float32, 1e-12, 1e-12)

    assert obj.krylov_tol == pytest.approx(tol_floor(np.float32))
    assert obj.newton_rtol == pytest.approx(tol_floor(np.float32))
    assert report['krylov_clamped'] is True
    assert report['newton_clamped'] is True
    assert report['working_precision'] == 'float32'
    assert 'clamped' in obj.describe_tolerances()

    obj64 = Dummy()
    obj64.setup_precision_tolerances(None, 1e-8, 1e-8)
    assert obj64.krylov_tol == pytest.approx(1e-8)
    assert obj64.solve_precision is None
    assert 'clamped' not in obj64.describe_tolerances()
