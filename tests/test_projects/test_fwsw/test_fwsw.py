import numpy as np
import os

from projects.FastWaveSlowWave.plot_stifflimit_specrad import main, plot_results

def test_stifflimit_specrad():
    nodes_v, lambda_f, specrad, norm = main()
    assert np.amax(specrad) < 0.9715, 'Spectral radius is too high, got %s' % specrad
    assert np.amax(norm) < 2.210096, 'Norm is too high, got %s' % norm

    plot_results(nodes_v, lambda_f, specrad, norm)
    assert os.path.isfile('stifflimit-specrad.png'), 'ERROR: specrad plot has not been created'
    assert os.path.isfile('stifflimit-norm.png'), 'ERROR: norm plot has not been created'
