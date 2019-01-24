import os

from pySDC.projects.FastWaveSlowWave.runconvergence_acoustic import plot_convergence
from pySDC.projects.FastWaveSlowWave.runitererror_acoustic import compute_and_plot_itererror
from pySDC.projects.FastWaveSlowWave.runmultiscale_acoustic import compute_and_plot_solutions


def test_plot_convergence():
    assert os.path.isfile('pySDC/projects/FastWaveSlowWave/data/conv-data.txt'), 'ERROR: conv-data.txt does not exist'
    plot_convergence(cwd='pySDC/projects/FastWaveSlowWave/')
    assert os.path.isfile('data/convergence.png'), 'ERROR: convergence plot has not been created'


def test_compute_and_plot_itererror():
    compute_and_plot_itererror()
    assert os.path.isfile('data/iteration.png'), 'ERROR: iteration plot has not been created'


def test_compute_and_plot_solutions():
    compute_and_plot_solutions()
    assert os.path.isfile('data/multiscale-K2-M2.png'), 'ERROR: solution plot has not been created'
