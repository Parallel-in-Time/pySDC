import pytest


@pytest.mark.base
@pytest.mark.parametrize('N', [4, 69])
def test_get_FFT_matrix(N):
    import numpy as np
    from pySDC.helpers.ParaDiagHelper import get_FFT_matrix

    fft_mat = get_FFT_matrix(N)

    data = np.random.random(N)

    fft1 = fft_mat @ data
    fft2 = np.fft.fft(data, norm='ortho')
    assert np.allclose(fft1, fft2), 'Forward transform incorrect'

    ifft1 = np.conjugate(fft_mat) @ data
    ifft2 = np.fft.ifft(data, norm='ortho')
    assert np.allclose(ifft1, ifft2), 'Backward transform incorrect'
