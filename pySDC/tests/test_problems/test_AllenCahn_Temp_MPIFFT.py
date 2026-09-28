import pytest


@pytest.mark.mpi4py
@pytest.mark.parametrize('spectral', [True, False])
def test_circle_rand(spectral):
    """
    ``circle_rand`` puts one circle of random radius in each of the L x L unit cells, so the phase field
    has to stay in [0, 1] and be close to 1 at every cell centre, and the temperature starts at 1.
    """
    import numpy as np
    from pySDC.implementations.problem_classes.AllenCahn_Temp_MPIFFT import allencahn_temp_imex

    L = 2
    prob = allencahn_temp_imex(nvars=(64, 64), L=L, init_type='circle_rand', spectral=spectral)
    u = prob.u_exact(0)
    phi = prob.fft.backward(u[..., 0]) if spectral else u[..., 0]
    temp = prob.fft.backward(u[..., 1]) if spectral else u[..., 1]

    assert np.all((phi > -1e-12) & (phi < 1 + 1e-12))
    assert np.allclose(temp, 1.0)

    x, y = (np.asarray(X) for X in prob.X)
    for cx in (-0.5, 0.5):
        for cy in (-0.5, 0.5):
            centre = np.argmin((x - cx) ** 2 + (y - cy) ** 2)
            assert phi.flat[centre] > 0.9, f'no circle in the cell centred at ({cx}, {cy})'
