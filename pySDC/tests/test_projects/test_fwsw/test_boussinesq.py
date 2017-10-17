import os

from pySDC.projects.FastWaveSlowWave.plotgmrescounter_boussinesq import plot_buoyancy


def test_plot():

    assert os.path.isfile('pySDC/projects/FastWaveSlowWave/data/xaxis.npy'), 'ERROR: xaxis.npy does not exist'
    assert os.path.isfile('pySDC/projects/FastWaveSlowWave/data/sdc.npy'), 'ERROR: sdc.npy does not exist'
    assert os.path.isfile('pySDC/projects/FastWaveSlowWave/data/dirk.npy'), 'ERROR: dirk.npy does not exist'
    assert os.path.isfile('pySDC/projects/FastWaveSlowWave/data/rkimex.npy'), 'ERROR: rkimex.npy does not exist'
    assert os.path.isfile('pySDC/projects/FastWaveSlowWave/data/uref.npy'), 'ERROR: uref.npy does not exist'
    assert os.path.isfile('pySDC/projects/FastWaveSlowWave/data/split.npy'), 'ERROR: split.npy does not exist'
    plot_buoyancy(cwd='pySDC/projects/FastWaveSlowWave/')
    assert os.path.isfile('data/boussinesq.png'), 'ERROR: buoyancy plot has not been created'
