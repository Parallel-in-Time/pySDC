import pytest


@pytest.mark.base
@pytest.mark.parametrize('error, expected', [(2e-16, 0), (1e-3, float('inf'))])
def test_no_contraction(error, expected):
    """
    An error that does not contract is converged already or never, and is not divided by log(1) = 0.
    """
    import warnings
    from types import SimpleNamespace
    from pySDC.implementations.convergence_controller_classes.estimate_contraction_factor import (
        EstimateContractionFactor,
    )

    ctrl = object.__new__(EstimateContractionFactor)
    ctrl.params = SimpleNamespace(e_tol=1e-7)
    status = SimpleNamespace(error_embedded_estimate=error, error_embedded_estimate_last_iter=error)
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        ctrl.post_iteration_processing(None, SimpleNamespace(levels=[SimpleNamespace(status=status)]))
    assert status.contraction_factor == 1
    assert status.iter_to_convergence == expected
