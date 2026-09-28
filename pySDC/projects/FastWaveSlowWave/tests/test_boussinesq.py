import os
import pytest


@pytest.mark.base
def test_plot():
    from pySDC.projects.FastWaveSlowWave.plotgmrescounter_boussinesq import plot_buoyancy

    assert os.path.isfile('pySDC/projects/FastWaveSlowWave/data/xaxis.npy'), 'ERROR: xaxis.npy does not exist'
    assert os.path.isfile('pySDC/projects/FastWaveSlowWave/data/sdc.npy'), 'ERROR: sdc.npy does not exist'
    assert os.path.isfile('pySDC/projects/FastWaveSlowWave/data/dirk.npy'), 'ERROR: dirk.npy does not exist'
    assert os.path.isfile('pySDC/projects/FastWaveSlowWave/data/rkimex.npy'), 'ERROR: rkimex.npy does not exist'
    assert os.path.isfile('pySDC/projects/FastWaveSlowWave/data/uref.npy'), 'ERROR: uref.npy does not exist'
    assert os.path.isfile('pySDC/projects/FastWaveSlowWave/data/split.npy'), 'ERROR: split.npy does not exist'
    plot_buoyancy(cwd='pySDC/projects/FastWaveSlowWave/')
    assert os.path.isfile('data/boussinesq.png'), 'ERROR: buoyancy plot has not been created'


@pytest.mark.base
def test_run(tmp_path):
    """
    Run the example itself, small enough to be cheap.

    The stored data is produced at the full resolution and this runs far too few steps to be
    accurate, so it cannot check numbers. What it does check is that every integrator in the
    script still runs at all and that none of them blows up, which is what `data/` was missing
    a test for: the script had been unrunnable for years without CI noticing.
    """
    import numpy as np
    from pySDC.projects.FastWaveSlowWave.rungmrescounter_boussinesq import main

    (tmp_path / 'data').mkdir()
    main(cwd=f'{tmp_path}/', nvars=(4, 30, 15), Tend=60, Nsteps=2)

    uref = np.load(tmp_path / 'data' / 'uref.npy')
    for name in ['sdc', 'dirk', 'rkimex', 'split']:
        u = np.load(tmp_path / 'data' / f'{name}.npy')
        assert np.all(np.isfinite(u)), f'ERROR: {name} did not stay finite'
        error = np.linalg.norm((u - uref).flatten(), np.inf) / np.linalg.norm(uref.flatten(), np.inf)
        assert error < 1.0, f'ERROR: {name} is off by {error:.3e}, it did not stay in the same ballpark'
