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
def test_clamp_none_selects_the_floor():
    """Passing no tolerance falls back to the floor for that precision and conditioning."""
    import numpy as np
    from pySDC.projects.DeltaSDC.precision import clamp_tolerance, tol_floor

    value, clamped = clamp_tolerance(None, np.float32, conditioning=500.0)
    assert value == pytest.approx(tol_floor(np.float32, conditioning=500.0)) and clamped is False
