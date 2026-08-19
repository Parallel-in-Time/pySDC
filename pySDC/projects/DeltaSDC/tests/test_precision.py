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


@pytest.mark.base
def test_tol_floor_is_conditioning_aware():
    """The floor must grow with alpha*||J||, which is what a fixed multiple of eps misses."""
    import numpy as np
    from pySDC.projects.DeltaSDC.precision import tol_floor, working_eps

    eps32 = working_eps(np.float32)

    # below the unconditional minimum the floor does not move
    assert tol_floor(np.float32, conditioning=1.0) == pytest.approx(100.0 * eps32)
    assert tol_floor(np.float32, conditioning=10.0) == pytest.approx(100.0 * eps32)

    # above it the floor scales with the conditioning
    assert tol_floor(np.float32, conditioning=200.0) == pytest.approx(4.0 * 200.0 * eps32)
    assert tol_floor(np.float32, conditioning=2000.0) == pytest.approx(10 * tol_floor(np.float32, conditioning=200.0))

    # the multipliers are configurable
    assert tol_floor(np.float32, safety=1.0, conditioning=1.0, conditioning_safety=1.0) == pytest.approx(eps32)


@pytest.mark.base
def test_clamp_tolerance_uses_conditioning():
    """A tolerance that is fine for a mild system must be clamped for a stiff one."""
    import numpy as np
    from pySDC.projects.DeltaSDC.precision import clamp_tolerance, tol_floor

    requested = 1e-4  # above the unconditional floor of 100*eps32 = 1.19e-05

    value, clamped = clamp_tolerance(requested, np.float32, conditioning=1.0)
    assert value == pytest.approx(requested) and clamped is False

    value, clamped = clamp_tolerance(requested, np.float32, conditioning=1000.0)
    assert value == pytest.approx(tol_floor(np.float32, conditioning=1000.0)) and clamped is True


@pytest.mark.base
def test_effective_tolerances_reclamps_for_conditioning():
    """The mixin must re-derive the tolerances once the conditioning is known."""
    import numpy as np
    from pySDC.projects.DeltaSDC.precision import PrecisionAwareTolerances, tol_floor

    class Dummy(PrecisionAwareTolerances):
        pass

    obj = Dummy()
    obj.setup_precision_tolerances(np.float32, 1e-4, 1e-4)

    mild_krylov, mild_newton = obj.effective_tolerances(1.0)
    assert mild_krylov == pytest.approx(1e-4) and mild_newton == pytest.approx(1e-4)

    stiff_krylov, stiff_newton = obj.effective_tolerances(1000.0)
    expected = tol_floor(np.float32, conditioning=1000.0)
    assert stiff_krylov == pytest.approx(expected) and stiff_newton == pytest.approx(expected)
    assert stiff_krylov > mild_krylov
