import numpy as np
import os

from projects.FastWaveSlowWave.plot_stifflimit_specrad import compute_specrad, plot_specrad
from projects.FastWaveSlowWave.plot_stability import compute_stability, plot_stability
from projects.FastWaveSlowWave.plot_stab_vs_k import compute_stab_vs_k, plot_stab_vs_k
from projects.FastWaveSlowWave.plot_dispersion import compute_and_plot_dispersion


def test_stifflimit_specrad():
    nodes_v, lambda_f, specrad, norm = compute_specrad()
    assert np.amax(specrad) < 0.9715, 'Spectral radius is too high, got %s' % specrad
    assert np.amax(norm) < 2.210096, 'Norm is too high, got %s' % norm

    plot_specrad(nodes_v, lambda_f, specrad, norm)
    assert os.path.isfile('stifflimit-specrad.png'), 'ERROR: specrad plot has not been created'
    assert os.path.isfile('stifflimit-norm.png'), 'ERROR: norm plot has not been created'


def test_stability():
    lambda_s, lambda_f, num_nodes, K, stab = compute_stability()
    assert np.amax(stab).real < 26.327931, "Real part of max. stability too large, got %s" % stab
    assert np.amax(stab).imag < 0.2467791, "Imag part of max. stability too large, got %s" % stab

    plot_stability(lambda_s, lambda_f, num_nodes, K, stab)
    assert os.path.isfile('stability-K3-M3.png'), 'ERROR: stability plot has not been created'


def test_stab_vs_k():
    mvals, kvals, stabval = compute_stab_vs_k(slow_resolved=True)
    assert np.amax(stabval) < 1.4455919, 'ERROR: stability values are too high, got %s' % stabval

    plot_stab_vs_k(True, mvals, kvals, stabval)
    assert os.path.isfile('stab_vs_k_resolved.png'), 'ERROR: stability plot has not been created'

    mvals, kvals, stabval = compute_stab_vs_k(slow_resolved=False)
    assert np.amax(stabval) < 3.7252282, 'ERROR: stability values are too high, got %s' % stabval

    plot_stab_vs_k(False, mvals, kvals, stabval)
    assert os.path.isfile('stab_vs_k_unresolved.png'), 'ERROR: stability plot has not been created'


def test_dispersion():
    compute_and_plot_dispersion()
    assert os.path.isfile('phase-K3-M3.png'), 'ERROR: phase plot has not been created'
    assert os.path.isfile('ampfactor-K3-M3.png'), 'ERROR: phase plot has not been created'
