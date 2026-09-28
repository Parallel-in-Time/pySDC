import pytest


@pytest.mark.base
@pytest.mark.parametrize(
    'ndim, freq, nvars',
    [
        (1, -1, (64, 128, 256)),
        (2, 2, (32, 64, 128)),
        (3, 2, (16, 32, 64)),
    ],
)
@pytest.mark.parametrize('order', [2, 4])
def test_u_exact_converges_in_space(ndim, freq, nvars, order):
    """
    u_exact must solve the PDE, so plugging it into the semi-discrete right-hand side leaves only the
    spatial discretization error, which has to fall at the stencil order.
    """
    import numpy as np
    from pySDC.implementations.problem_classes.AdvectionEquation_ND_FD import advectionNd
    from pySDC.tests.test_problems.test_heatNd_FD import spatial_residual

    errors = [
        spatial_residual(advectionNd(nvars=(n,) * ndim, freq=(freq,) * ndim, c=1.0, order=order, bc='periodic'), 0.3)
        for n in nvars
    ]
    rates = np.log2(np.array(errors[:-1]) / np.array(errors[1:]))
    assert np.all(rates > order - 0.3), f'expected order {order}, got {rates} from errors {errors}'
