import os
import pytest
import numpy as np


@pytest.mark.base
def test_stifflimit_specrad():
    from pySDC.projects.FastWaveSlowWave.plot_stifflimit_specrad import compute_specrad, plot_specrad

    nodes_v, lambda_f, specrad, norm = compute_specrad()
    assert np.amax(specrad) < 0.9715, 'Spectral radius is too high, got %s' % specrad
    assert np.amax(norm) < 2.210096, 'Norm is too high, got %s' % norm

    plot_specrad(nodes_v, lambda_f, specrad, norm)
    assert os.path.isfile('data/stifflimit-specrad.png'), 'ERROR: specrad plot has not been created'
    assert os.path.isfile('data/stifflimit-norm.png'), 'ERROR: norm plot has not been created'


@pytest.mark.base
def test_stability():
    from pySDC.projects.FastWaveSlowWave.plot_stability import compute_stability, plot_stability

    lambda_s, lambda_f, num_nodes, K, stab = compute_stability()
    assert np.amax(stab).real < 26.327931, "Real part of max. stability too large, got %s" % stab
    assert np.amax(stab).imag < 0.2467791, "Imag part of max. stability too large, got %s" % stab

    plot_stability(lambda_s, lambda_f, num_nodes, K, stab)
    assert os.path.isfile('data/stability-K3-M3.png'), 'ERROR: stability plot has not been created'


@pytest.mark.base
def test_stab_vs_k():
    from pySDC.projects.FastWaveSlowWave.plot_stab_vs_k import compute_stab_vs_k, plot_stab_vs_k

    mvals, kvals, stabval = compute_stab_vs_k(slow_resolved=True)
    assert np.amax(stabval) < 1.4455919, 'ERROR: stability values are too high, got %s' % stabval

    plot_stab_vs_k(True, mvals, kvals, stabval)
    assert os.path.isfile('data/stab_vs_k_resolved.png'), 'ERROR: stability plot has not been created'

    mvals, kvals, stabval = compute_stab_vs_k(slow_resolved=False)
    assert np.amax(stabval) < 3.7252282, 'ERROR: stability values are too high, got %s' % stabval

    plot_stab_vs_k(False, mvals, kvals, stabval)
    assert os.path.isfile('data/stab_vs_k_unresolved.png'), 'ERROR: stability plot has not been created'


@pytest.mark.base
def test_dispersion():
    from pySDC.projects.FastWaveSlowWave.plot_dispersion import compute_and_plot_dispersion

    compute_and_plot_dispersion()
    assert os.path.isfile('data/phase-K3-M3.png'), 'ERROR: phase plot has not been created'
    assert os.path.isfile('data/ampfactor-K3-M3.png'), 'ERROR: phase plot has not been created'

    compute_and_plot_dispersion(Nsamples=3, K=4)
    compute_and_plot_dispersion(Nsamples=3, K=5)


@pytest.mark.base
@pytest.mark.parametrize('K', [2, 3, 5])
@pytest.mark.parametrize('num_nodes', [2, 3])
def test_system_stability_matches_scalar(K, num_nodes):
    """
    The acoustic-advection stability matrix used for the dispersion plots has to be the scalar IMEX-SDC
    stability function of imex_1st_order, applied mode by mode: Uadv is a multiple of the identity, so it
    commutes with Cs and both diagonalize in the eigenvectors of Cs. The scalar K-sweep matrix is itself
    checked against actual sweeps in pySDC/tests/test_sweepers/test_imexsweeper.py.
    """
    from pySDC.core.step import Step
    from pySDC.implementations.problem_classes.FastWaveSlowWave_0D import swfw_scalar
    from pySDC.implementations.sweeper_classes.imex_1st_order import imex_1st_order
    from pySDC.projects.FastWaveSlowWave.plot_dispersion import sdc_system_stability

    description = {
        'problem_class': swfw_scalar,
        'problem_params': {'lambda_s': np.array([0.0]), 'lambda_f': np.array([0.0]), 'u0': 1.0},
        'sweeper_class': imex_1st_order,
        'sweeper_params': {'quad_type': 'RADAU-RIGHT', 'do_coll_update': True, 'num_nodes': num_nodes},
        'level_params': {'dt': 1.0},
        'step_params': {},
    }
    L = Step(description=description).levels[0]
    weights, ones = L.sweep.coll.weights, np.ones(num_nodes)

    for k in np.linspace(0, np.pi, 6)[1:]:
        Cs = -1j * k * np.array([[0.0, 1.0], [1.0, 0.0]])
        lam_slow = -1j * k * 0.05
        stab = sdc_system_stability(L, Cs, lam_slow * np.eye(2), K)

        lam_fast, V = np.linalg.eig(Cs)
        R = [
            1 + (lf + lam_slow) * weights @ L.sweep.get_scalar_problems_manysweep_mat(K, [lf, lam_slow]) @ ones
            for lf in lam_fast
        ]
        assert np.allclose(stab, V @ np.diag(R) @ np.linalg.inv(V), atol=1e-13, rtol=0)
